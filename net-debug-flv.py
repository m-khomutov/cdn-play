from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from enum import IntEnum
import argparse
import asyncio
import av
import base64
import cv2
import datetime
import hashlib
import numpy as np
import pyaudio
import queue
import requests
import ssl
import struct
import threading
import urllib.parse

class HashManager:
    def __init__(self, **argv):
        self._set_password_hash(argv.get('password'))
        self._set_control_url(argv.get('hash'))

    def _set_password_hash(self, password):
        self._password_hash=None
        if password:
            sha1_hash = hashlib.sha1()
            sha1_hash.update(password.encode('utf-8'))
            self._password_hash=sha1_hash.digest()[0:16]

    def _set_control_url(self, hashed):
        self._control_url=None
        self._cipher=None
        self._unencrypted=''
        if self._password_hash and hashed:
            lst = hashed.lstrip('/').split('/')
            hashed = lst[0]
            self._unencrypted = '/'.join(lst[1:])
            while len(hashed) % 8:
                hashed = hashed+'='
            encoded = base64.b32decode(hashed.encode('utf-8'))
            self._cipher = Cipher(algorithms.AES(self._password_hash), modes.ECB(), backend=default_backend())
            decryptor=self._cipher.decryptor()
            padded = decryptor.update(encoded) + decryptor.finalize()
            pad_remover = padding.PKCS7(algorithms.AES.block_size).unpadder()
            params = (pad_remover.update(padded) + pad_remover.finalize()).decode().split('/')
            self._control_url= '/'.join((params[0],
                                         params[1],
                                         params[2]+'//'+params[4].replace('1232', '2232'),
                                         '?control='+params[-3]+'&action='))

    def _get_control_url_hash(self, url):
        if self._cipher:
            pad_inserter = padding.PKCS7(algorithms.AES.block_size).padder()
            padded = pad_inserter.update(url.encode('utf-8')) + pad_inserter.finalize()
            encryptor=self._cipher.encryptor()
            return base64.b32encode(encryptor.update(padded)+encryptor.finalize()).decode('utf-8').rstrip('=')
        return None

    def get_control_url(self, **kwargs):
        if self._control_url:
            control_url=self._control_url
            action=kwargs.get('action', None)
            if action:
                control_url=control_url+action
            pos=kwargs.get('pos', None)
            if pos:
                control_url=control_url+'&pos='+pos
            scale=kwargs.get('scale', None)
            if scale:
                control_url=control_url+'&scale='+scale
            print(f'command: {control_url}')
            if hashed := self._get_control_url_hash(control_url):
                hashed='/'+hashed
                if self._unencrypted:
                    return hashed+'/' + self._unencrypted
                return hashed
        return None


class AudioSpecificConfig:
    def __init__(self, data):
        self.object_type=1
        self.frequency_index=4
        self.channel_config=1
        if len(data) >= 2:
            # 5 bits: object type
            # if (object type == 31)
            # 6 bits + 32: object type
            self.object_type = int(data[0]) >> 3
            # 4 bits: frequency index
            # if (frequency index == 15)
            # 24 bits: frequency
            self.frequency_index = ((int(data[0]) & 7) << 1) | (int(data[1]) >> 7)
            # 4 bits: channel configuration
            self.channel_config = (int(data[1]) >> 3) & 15

    def generate_adts_header(self, sample_size):
        adts_header=b'\xff\xf1'
        sample_length=len(sample_size)+7
        adts_header=adts_header+((((self.object_type-1) & 3) << 6) | ((self.frequency_index & 15) << 2)).to_bytes(1, 'big')
        adts_header=adts_header+((self.channel_config << 6) | ((sample_length >> 11) & 3)).to_bytes(1, 'big')
        adts_header=adts_header+((sample_length >> 3) & 0xff).to_bytes(1, 'big')
        adts_header=adts_header+(((sample_length & 7) << 5) | 0x1f).to_bytes(1, 'big')
        return adts_header+b'\xfc'

    def frequency(self):
        if self.frequency_index == 0:
            return 96000
        elif self.frequency_index == 1:
            return 88200
        elif self.frequency_index == 2:
            return 64000
        elif self.frequency_index == 3:
            return 48000
        elif self.frequency_index == 4:
            return 44100
        elif self.frequency_index == 5:
            return 32000
        elif self.frequency_index == 6:
            return 24000
        elif self.frequency_index == 7:
            return 22050
        elif self.frequency_index == 8:
            return 16000
        elif self.frequency_index == 9:
            return 12000
        elif self.frequency_index == 10:
            return 11025
        return 8000


class Slider:
    def __init__(self):
        self._thickness = 2
        self._passed_color = (255, 0, 0) # blue
        self._left_color = (194, 153, 35)
        self._marker_color = (0, 0, 255) # red
        self._marker_radius = 2
        self._offset = 20
        self._y=0

    def draw(self, image, image_size, archive_position, archive_range, hint):
        self._y=image_size[1]-self._offset
        image_start_point=(self._offset, self._y)
        image_end_point=(image_size[0]-self._offset, self._y)
        coef=(image_end_point[0]-image_start_point[0])/(archive_range[1]-archive_range[0])
        position=(int((archive_position - archive_range[0]) * coef + image_start_point[0]), self._y)
        cv2.line(image, image_start_point, position, self._passed_color, self._thickness)
        cv2.line(image, position, image_end_point, self._left_color, self._thickness)
        cv2.circle(image, position, self._marker_radius, self._marker_color, self._thickness)
        ts = datetime.datetime.fromtimestamp(archive_range[0]).strftime("%d-%m-%Y %H:%M:%S")
        cv2.putText(image, ts, (0,self._y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
        ts = datetime.datetime.fromtimestamp(archive_range[1]).strftime("%d-%m-%Y %H:%M:%S")
        cv2.putText(image, ts, (image_size[0]-150,self._y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
        if hint and hint[2]:
            ts = datetime.datetime.fromtimestamp(int(hint[2])).strftime("%d-%m-%Y %H:%M:%S")
            cv2.putText(image, ts, (hint[0],self._y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    def calculate_position(self, image_size, image_position, archive_range):
        if abs(image_position[1]-self._y) < 5:
            coef = (archive_range[1] - archive_range[0]) / (image_size[0] - 2 * self._offset)
            return int(image_position[0] - self._offset) * coef + archive_range[0]
        return None


class ButtonGroup:
    def __init__(self):
        self._thickness = 1
        self._pressed_color = (157, 157, 231)  # pink
        self._border_color = (0, 0, 255)  # red
        self._offset = 15
        self._half_button_size = 10

    def draw(self, image, image_size):
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def click_on_button(click_pos, lt, rb):
        if click_pos[0] >= lt[0] and click_pos[0] <= rb[0]:
            return click_pos[1] >= lt[1] and click_pos[1] <= rb[1]
        return False


class RadioButton:
    def __init__(self, **kwargs):
        self.caption=kwargs.get('caption', '1')
        self.lt=kwargs.get('lt', None)
        self.rb=kwargs.get('rb', None)
        self.toggled=kwargs.get('toggled', False)


class RadioButtonGroup(ButtonGroup):
    def __init__(self):
        super().__init__()
        self._buttons=[RadioButton(caption='1',toggled=True),
                       RadioButton(caption='2'),
                       RadioButton(caption='4'),
                       RadioButton(caption='8'),
                       RadioButton(caption='16'),
                       RadioButton(caption='32')]
        self._position=None

    def draw(self, image, image_size):
        self._position=((image_size[0] // 3, image_size[1]-self._offset),
                        (image_size[0] // 3 + 2 * len(self._buttons) * self._half_button_size, image_size[1]-1))
        lt = self._position[0]
        cv2.putText(image,
                    'scale:',
                    (lt[0]-60,lt[1]+self._half_button_size), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self._border_color,
                    self._thickness,
                    cv2.LINE_AA)
        for btn in self._buttons:
            color=self._pressed_color if btn.toggled else self._border_color
            btn.lt=lt
            btn.rb=(lt[0]+2*self._half_button_size, lt[1]+self._half_button_size+2)
            cv2.rectangle(image, btn.lt, btn.rb, color, self._thickness)
            cv2.putText(image,
                        btn.caption,
                        (lt[0]+self._half_button_size//2, lt[1]+self._half_button_size),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        self._thickness,
                        cv2.LINE_AA)
            lt=(lt[0]+2*self._half_button_size+2, lt[1])

    def click(self, click_pos):
        caption=None
        for btn in self._buttons:
            btn.toggled=False
            if btn.lt and btn.rb:
                if PushButtonGroup.click_on_button(click_pos,btn.lt,btn.rb):
                    btn.toggled=True
                    caption=btn.caption
        return caption


class PushButton(IntEnum):
    PLAY=0,
    REVERSE_PLAY=1,
    PAUSE=2


class PushButtonGroup(ButtonGroup):
    def __init__(self):
        super().__init__()
        self._pause_button_position=None
        self._play_button_position=None
        self._reverse_play_button_position=None
        self._pressed_button=PushButton.PLAY

    def draw(self, image, image_size):
        self._pause_button_position=((image_size[0] // 2 - self._half_button_size, image_size[1]-self._offset),
                                     (image_size[0] // 2 + self._half_button_size, image_size[1]-1))
        self._draw_pause_button(image)
        self._play_button_position=((image_size[0] // 2 + 2*self._half_button_size, image_size[1]-self._offset),
                                    (image_size[0] // 2 + 4*self._half_button_size, image_size[1]-1))
        self._draw_play_button(image)
        self._reverse_play_button_position=((image_size[0] // 2 - 4*self._half_button_size, image_size[1] - self._offset),
                                            (image_size[0] // 2 - 2* self._half_button_size, image_size[1] - 1))
        self._draw_reverse_play_button(image)

    def click_pause_button(self, click_pos):
        if self._pause_button_position:
            if PushButtonGroup.click_on_button(click_pos,self._pause_button_position[0],self._pause_button_position[1]):
                self._pressed_button=PushButton.PAUSE
                return True
        return False

    def click_play_button(self, click_pos):
        if self._play_button_position:
            if PushButtonGroup.click_on_button(click_pos,self._play_button_position[0],self._play_button_position[1]):
                self._pressed_button=PushButton.PLAY
                return True
        return False

    def click_reverse_play_button(self, click_pos):
        if self._reverse_play_button_position:
            if PushButtonGroup.click_on_button(click_pos,self._reverse_play_button_position[0],self._reverse_play_button_position[1]):
                self._pressed_button=PushButton.REVERSE_PLAY
                return True
        return False

    def _draw_pause_button(self, image):
        color=self._pressed_color if self._pressed_button == PushButton.PAUSE else self._border_color
        # граница
        left_top=self._pause_button_position[0]
        right_bottom=self._pause_button_position[1]
        cv2.rectangle(image, left_top, right_bottom, color, self._thickness)
        # левая вертикальная полоса
        lt=(left_top[0]+7,left_top[1]+3)
        rb=(left_top[0]+7, right_bottom[1]-3)
        cv2.line(image, lt, rb, color, self._thickness*3)
        # правая вертикальная полоса
        lt=(right_bottom[0]-7,left_top[1]+3)
        rb=(right_bottom[0]-7,right_bottom[1]-3)
        cv2.line(image, lt, rb, color, self._thickness*3)

    def _draw_play_button(self, image):
        color=self._pressed_color if self._pressed_button == PushButton.PLAY else self._border_color
        left_top=self._play_button_position[0]
        right_bottom=self._play_button_position[1]
        # граница
        cv2.rectangle(image, left_top, right_bottom, color, self._thickness)
        # вертикальная
        lt=(left_top[0]+5,left_top[1]+3)
        rb=(left_top[0]+5, right_bottom[1]-3)
        cv2.line(image, lt, rb, color, self._thickness*2)
        # LT - RMiddle
        lt=(left_top[0]+5,left_top[1]+3)
        rm=(right_bottom[0]-5,right_bottom[1]-7)
        cv2.line(image, lt, rm, color, self._thickness*2)
        # RMiddle - LB
        rm = (right_bottom[0] - 5, right_bottom[1] - 7)
        lb = (left_top[0] + 5, right_bottom[1] - 3)
        cv2.line(image, rm, lb, color, self._thickness * 2)

    def _draw_reverse_play_button(self, image):
        color=self._pressed_color if self._pressed_button == PushButton.REVERSE_PLAY else self._border_color
        left_top=self._reverse_play_button_position[0]
        right_bottom=self._reverse_play_button_position[1]
        # граница
        cv2.rectangle(image, left_top, right_bottom, color, self._thickness)
        # вертикальная
        rt = (right_bottom[0] - 5, left_top[1] + 3)
        rb = (right_bottom[0] - 5, right_bottom[1] - 3)
        cv2.line(image, rt, rb, color, self._thickness * 2)
        # RT - LMiddle
        rt = (right_bottom[0] - 5, left_top[1] + 3)
        lm = (left_top[0] + 5, right_bottom[1] - 7)
        cv2.line(image, rt, lm, color, self._thickness * 2)
        # LMiddle - RB
        lm = (left_top[0] + 5, right_bottom[1] - 7)
        rb = (right_bottom[0] - 5, right_bottom[1] - 3)
        cv2.line(image, lm, rb, color, self._thickness * 2)


class Renderer(threading.Thread):
    def __init__(self, **argv):
        super(Renderer, self).__init__()
        self._caption = argv.get('caption', 'stream debug')
        self._address = argv.get('address')
        self._slider_caption = 'timestamp slider'
        self._data_queue = argv.get('queue')
        self._terminated_flag = argv.get('terminated')
        self._font_face = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.8
        self._font_color = (0, 0, 255)  # Red color in BGR
        self._font_thickness = 2
        self._line_type = cv2.LINE_AA
        self._match_size_flag = True
        self._timestamp = None
        self._sei_timestamp = None
        self._position = None
        self._sei_position = None
        self._archive_range = [0,0]
        self._hint = None
        try:
            self._hash_manager = HashManager(password=argv.get('password'),
                                             hash=argv.get('caption'))
        except ValueError as ve:
            print(f'failed to make HashManager: {ve}')
            self._hash_manager=None
        self._slider=Slider()
        self._push_buttons=PushButtonGroup()
        self._scale_buttons=RadioButtonGroup()
        self._audio_specific_config=None
        self._pyaudio=None
        self._audio_stream=None
        self._array=np.zeros((480, 640, 3), dtype=np.uint8)

    def __del__(self):
        if self._pyaudio:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
            self._pyaudio.terminate()
        cv2.destroyAllWindows()

    def run(self):
        current_datetime = datetime.datetime.now()
        cv2.namedWindow(self._caption, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self._caption, self._on_mouse_event)
        while not self._terminated_flag.is_set():
            try:
                if cv2.getWindowProperty(self._caption, cv2.WND_PROP_VISIBLE) < 1:
                    self._terminated_flag.set()
                    break
            except:
                self._terminated_flag.set()
                break
            try:
                video,self._array,sei_timestamp=self._data_queue.get_nowait()
                if video:
                    if sei_timestamp:
                        self._sei_position=sei_timestamp//1000
                        self._sei_timestamp=datetime.datetime.fromtimestamp(self._sei_position).strftime("%d-%m-%Y %H:%M:%S")
                    self._draw_frame(self._array)
                else:
                    self._play_packet(self._array)
                delta = datetime.datetime.now() - current_datetime
                if delta.total_seconds() >= 1:
                    current_datetime = datetime.datetime.now()
                    self._get_position()
            except:
                if self._archive_range[0]:
                    x, y, w, h = cv2.getWindowImageRect(self._caption)
                    self._push_buttons.draw(self._array, (w, h))
                    self._scale_buttons.draw(self._array, (w, h))
                try:
                    cv2.imshow(self._caption, self._array)
                except cv2.error as err:
                    print(f'opencv2 error: {err}')
                cv2.waitKey(10)

    def move_to_timestamp(self, timestamp):
        self._request_action(action='seek', pos=f'{timestamp}')

    def scale(self, coefficient):
        self._request_action(action='scale', pos=f'{coefficient}')

    def play(self):
        self._request_action(action='play')

    def reverse_play(self):
        self._request_action(action='rplay')

    def pause(self):
        self._request_action(action='pause')

    def _request_action(self, **kwargs):
        if not self._hash_manager:
            return None
        response=None
        try:
            hashed=self._hash_manager.get_control_url(**kwargs)
            if hashed:
                url=self._address+hashed
                response=requests.get(url, params={}, headers={})
                response.raise_for_status()
                print(response.json()) # For JSON content
        except requests.exceptions.RequestException as request_exc:
            print(f"An error occurred: {request_exc}")
        return response

    def _draw_frame(self, array):
        x, y, w, h = cv2.getWindowImageRect(self._caption)
        if self._match_size_flag:
            if w != array.shape[1] or h != array.shape[0]:
                cv2.resizeWindow(self._caption, array.shape[1], array.shape[0])
            self._match_size_flag = False
        elif w != array.shape[1] or h != array.shape[0]:
            array = cv2.resize(array, (w, h), interpolation=cv2.INTER_AREA)
        timestamp=self._sei_timestamp if self._sei_timestamp else self._timestamp
        position=self._sei_position if self._sei_position else self._position
        if timestamp:
            cv2.putText(array,
                        timestamp,
                        (int(w / 16), int(h / 6)),
                        self._font_face,
                        self._font_scale,
                        self._font_color,
                        self._font_thickness,
                        self._line_type)
        if self._archive_range[0] and timestamp:
            self._slider.draw(array, (w, h), position, self._archive_range, self._hint)
            self._push_buttons.draw(array, (w, h))
            self._scale_buttons.draw(array, (w, h))
        cv2.imshow(self._caption, array)
        cv2.waitKey(10)

    def _play_packet(self, array):
        if array.dtype != 'float32':
            self._audio_specific_config=AudioSpecificConfig(array.tobytes())
            self._pyaudio = pyaudio.PyAudio()
            self._audio_stream = self._pyaudio.open(format=pyaudio.paFloat32,
                                                    channels=self._audio_specific_config.channel_config,
                                                    rate=self._audio_specific_config.frequency(),
                                                    output=True,
                                                    frames_per_buffer=1024)
        elif self._audio_stream:
            self._audio_stream.write(array.tobytes())

    def _get_position(self):
        response=self._request_action(action='getpos')
        if response:
            if pos := response.json()['position']:
                self._position = int(pos)
                self._timestamp = datetime.datetime.fromtimestamp(int(pos)).strftime("%d-%m-%Y %H:%M:%S")
            if pos := response.json()['start']:
                self._archive_range[0] = int(pos)
            if pos := response.json()['end']:
                self._archive_range[1] = int(pos)

    def _on_mouse_event(self, event, x, y, _, __):
        if event == cv2.EVENT_LBUTTONDOWN:
            _, _, w, h = cv2.getWindowImageRect(self._caption)
            pos = self._slider.calculate_position( (w, h), (x,y), self._archive_range)
            if pos:
                self.move_to_timestamp(pos)
            elif self._push_buttons.click_pause_button((x,y)):
                self.pause()
            elif self._push_buttons.click_play_button((x,y)):
                self.play()
            elif self._push_buttons.click_reverse_play_button((x,y)):
                self.reverse_play()
            else:
                if scale := self._scale_buttons.click((x,y)):
                    self.scale(scale)

        if event == cv2.EVENT_MOUSEMOVE:
            _, _, w, h = cv2.getWindowImageRect(self._caption)
            self._hint = x, y, self._slider.calculate_position((w, h), (x, y), self._archive_range)


class DumpAvc:
    def __init__(self, **argv):
        self._video_codec=av.CodecContext.create("h264", mode="r")
        self._video_codec.open()
        self._play_audio=argv.get('audio')
        self._audio_codec=av.CodecContext.create("aac", mode="r") if self._play_audio else None
        self._audio_specific_config=None
        self._data=b''
        self.sei_timestamp=None
        self._data_queue=queue.Queue()
        if self._play_audio:
            self._audio_codec.open()
        self._renderer = Renderer(caption=argv.get('path'),
                                  address=argv.get('address'),
                                  queue=self._data_queue,
                                  password=argv.get('password'),
                                  terminated=argv.get('terminated'))
        self._renderer.start()

    def __del__(self):
        self._renderer.join()

    def dump(self, data):
        self._data=self._data+b'\x00\x00\x00\x01'+data

    def decode(self):
        try:
            frames=self._video_codec.decode(av.packet.Packet(self._data))
            for frame in frames:
                self._data_queue.put((True,frame.to_ndarray(format='bgr24'),self.sei_timestamp))
        except av.error.InvalidDataError as err:
            print(err)
        self._data=b''

    def decode_audio(self, is_seq_header, packet):
        if self._play_audio:
            if is_seq_header:
                self._audio_specific_config=AudioSpecificConfig(packet)
                self._data_queue.put((False,np.array(packet),None))
            elif self._audio_specific_config:
                try:
                    if not DumpAvc.with_adts(packet):
                        packet=self._audio_specific_config.generate_adts_header(packet)+packet
                    for c in packet[0:9]:
                        print(f'{hex(c)}', end=' ')
                    print('')
                    frames=self._audio_codec.decode(av.packet.Packet(packet))
                    for frame in frames:
                        channels = frame.to_ndarray()
                        if len(channels) == 2: # стерео надо переложить в interleaved формат
                            interleaved = np.empty(2*channels[0].size, dtype=channels[0].dtype)
                            interleaved[0::2]=channels[0]
                            interleaved[1::2]=channels[1]
                            self._data_queue.put((False,interleaved,None))
                        else:
                            self._data_queue.put((False,channels,None))
                except  av.error.InvalidDataError as err:
                    print(f'audio error: {err}')

    @staticmethod
    def with_adts(packet):
        return packet[0]==0xff and (packet[1] & 0xf0)==0xf0


class Header:
    size = 9
    def __init__(self, buffer):
        if chr(buffer[0]) != 'F' or chr(buffer[1]) != 'L' or chr(buffer[2]) != 'V' or buffer[3] != 1:
            raise SyntaxError('Invalid FLV header')
        self.signature = chr(buffer[0])+chr(buffer[1])+chr(buffer[2])

    def __repr__(self):
        return f'signature: {self.signature}'


class TagSize:
    size = 4
    def __init__(self, buffer):
        if not buffer:
            raise EOFError
        self.value = struct.unpack('>I', buffer)[0]

    def __repr__(self):
        return f'{self.value}'


class Tag:
    size = 11
    def __init__(self, buffer):
        if not buffer:
            raise EOFError
        self.type = int(buffer[0])
        temp = b'\x00'+buffer[1:4]
        self.data_size = struct.unpack('>I', temp)[0]
        temp = buffer[7:8]+buffer[4:7]
        self.timestamp = struct.unpack('>I', temp)[0]

    def __repr__(self):
        return f'type: {self.type}; size: {self.data_size}; timestamp: {self.timestamp}'


class VideoTag:
    size = 1
    def __init__(self, buffer):
        if not buffer:
            raise EOFError
        value_ = int(buffer[0])
        self.frame_type = (value_ & 0xF0) >> 4
        self.codec_id = value_ & 0x0F

    def __repr__(self):
        return 'keyframe' if self.frame_type == 1 else 'interframe'


class AvcPacket:
    size = 4
    def __init__(self, buffer):
        if not buffer:
            raise EOFError
        self.packet_type = int(buffer[0])

    def __repr__(self):
        return 'AVC sequence header' if self.packet_type == 0 else 'AVC NALU'


class AudioData:
    size = 1
    def __init__(self, buffer):
        if not buffer:
            raise EOFError
        value_ = int(buffer[0])
        self.sound_format = (value_ >> 4) & 0x0F
        self.sound_rate = (value_ >> 6) & 3
        self.sound_size = (value_ >> 1) & 1
        self.sound_type = value_  & 1

    def __repr__(self):
        return 'AAC' if self.sound_format == 10 else str(self.sound_format)


class CDNHeader:
    size=40
    def __init__(self, buffer):
        (self.size,
         self.ts,
         self.ats,
         self.st,
         self.start,
         self.end,
         self.stream_count) = struct.unpack('<IQQHQQH', buffer) # noqa

    def __repr__(self):
        return f'archive range({self.start}, {self.end})'


class CDNStreamType:
    def __init__(self, value):
        self._description=['Unknown','H264','G711','AAC','config','mjpeg']
        self._value=value

    def __repr__(self):
        return self._description[self._value] if self._value < len(self._description) else self._description[0]

class CDNStreamHeader:
    size=22
    def __init__(self, buffer):
        (self.st_size,
         self.st_ts,
         self.st_ats,
         self.st_type) = struct.unpack('<IQQH', buffer) # noqa

    def __repr__(self):
        return f'packet {str(CDNStreamType(self.st_type))}; ts={self.st_ts}; ats={self.st_ats}'


class Sei:
    prev_timestamp=0

    @classmethod
    def get_diff(cls, timestamp):
        if not cls.prev_timestamp:
            cls.prev_timestamp=timestamp
        diff=timestamp-cls.prev_timestamp
        cls.prev_timestamp=timestamp
        return diff

    def __init__(self, buffer):
        self.type,self.size,self.timestamp = struct.unpack('<BBQ', buffer)

    def __repr__(self):
        return f'sei: type={self.type}; ts={self.timestamp} UTC={datetime.datetime.fromtimestamp(int(self.timestamp/1000))}'


class Sps:
    def __init__(self, nalu_type, buffer):
        self.data = hex(nalu_type) + ' '
        for x in buffer:
            self.data += hex(int(x)) + ' '

    def __repr__(self):
        return f'sps: {self.data}'


class Pps:
    def __init__(self, nalu_type, buffer):
        self.data = hex(nalu_type) + ' '
        for x in buffer:
            self.data += hex(int(x)) + ' '

    def __repr__(self):
        return f'pps: {self.data}'


async def read_bytes(reader, size):
    res = b''
    while len(res) < size:
        temp = await reader.read(size - len(res))
        if temp:
            res += temp
    return res


async def read_http_headers(reader):
    end_sequence = 0
    ret = bytearray()
    while True:
        c = await reader.read(1)
        ret+=c
        if c:
            if not chr(c[0]).isprintable() and c[0] != 13 and c[0] != 10:
                return False,ret
            if int(c[0]) == 13 or int(c[0]) == 10:
                end_sequence += 1
            else:
                end_sequence = 0
        if end_sequence == 4:
            break
    print(ret.decode("utf-8"))
    return True,ret


async def read_flv_header(reader):
    buf = await read_bytes(reader, Header.size)
    if not buf:
        raise EOFError
    print(str(Header(buf)))


async def read_flv_tag(reader):
    await read_bytes(reader, TagSize.size)
    buf = await read_bytes(reader, Tag.size)
    return Tag(buf)


async def read_flv_video_tag(reader):
    buf = await read_bytes(reader, VideoTag.size)
    return VideoTag(buf)


async def read_flv_avc_packet(reader):
    buf = await read_bytes(reader, AvcPacket.size)
    return AvcPacket(buf)


async def read_flv_audio_data(reader):
    buf = await read_bytes(reader, AudioData.size)
    return AudioData(buf)

async def read_flv_unit(reader, data_size):
    await read_bytes(reader, data_size)


async def read_video_flv_unit(reader, avc_dumper, tag, packet_type):
    offset = AvcPacket.size
    if packet_type == 1:
        while tag.data_size - VideoTag.size - offset > 0:
            buf = await read_bytes(reader, 5)  # size avcC
            sz,nalu_type = struct.unpack('>IB', buf)
            offset+=5
            if (nalu_type & 0x1f) == 6:
                buf = await read_bytes(reader, sz-1)  # type size timestamp
                try:
                    sei=Sei(buf[0:10])
                    print(f'{str(sei)}; diff: {sei.get_diff(sei.timestamp)}')
                    if avc_dumper:
                        avc_dumper.sei_timestamp = sei.timestamp
                except:
                    print('unknown SEI:', end=' ')
                    for x in buf:
                        print(f'{hex(x)}', end=' ')
                    print('')
                offset+=sz-1
            elif (nalu_type & 0x1f) == 7:
                buf = await read_bytes(reader, sz-1)  # type size timestamp
                if avc_dumper:
                    avc_dumper.dump(nalu_type.to_bytes(1, 'big')+buf)
                print(Sps(nalu_type, buf))
                offset+=sz-1
            elif (nalu_type & 0x1f) == 8:
                buf = await read_bytes(reader, sz-1)  # type size timestamp
                if avc_dumper:
                    avc_dumper.dump(nalu_type.to_bytes(1, 'big')+buf)
                print(Pps(nalu_type, buf))
                offset+=sz-1
            else:
                buf = await read_bytes(reader, sz-1)
                if avc_dumper:
                    avc_dumper.dump(nalu_type.to_bytes(1, 'big')+buf)
                    avc_dumper.decode()
                offset+=sz-1
    payload_size = tag.data_size - VideoTag.size - offset
    if payload_size > 0:
        await read_bytes(reader, payload_size)


async def read_flv(**argv):
    reader=argv.get('reader')
    terminated_flag=argv.get('terminated')
    avc_dumper=argv.get('dumper')

    await read_flv_header(reader)

    timestamp = dict()
    dt = datetime.datetime.now()
    while not terminated_flag.is_set():
        try:
            tag = await read_flv_tag(reader)
            if not tag.type in timestamp:
                timestamp[tag.type] = tag.timestamp
            if tag.type != 9:
                diff_ts = tag.timestamp - timestamp[tag.type]
                left_bytes=tag.data_size
                if tag.type == 8:
                    a_data = await read_flv_audio_data(reader)
                    left_bytes=left_bytes-AudioData.size
                    if a_data.sound_format==10:
                        buf=await read_bytes(reader, 1)
                        aac_packet_type=buf[0]
                        left_bytes=left_bytes-1
                        print(f'{str(a_data)} {"sequence_header" if aac_packet_type==0 else "raw"}: ', end='')
                        buf=await read_bytes(reader, left_bytes)
                        if aac_packet_type==0:
                            print('{', end=' ')
                            for b in buf:
                                print(f'{hex(b)}', end=' ')
                            print('}', end=' ')
                        if avc_dumper:
                            avc_dumper.decode_audio(aac_packet_type==0, buf)
                        left_bytes=0
                    else:
                        print(f'{str(a_data)}: ', end='')
                    print(f'timestamp: {tag.timestamp}; diff(ms): [ts: {diff_ts}] size: {tag.data_size}')
                timestamp[tag.type] = tag.timestamp
                if left_bytes:
                    await read_bytes(reader, left_bytes)
                continue
            video_tag = await read_flv_video_tag(reader)
            diff_ts = tag.timestamp - timestamp[tag.type]
            if diff_ts < 0:
                diff_ts = 0
            diff_dt = (datetime.datetime.now().timestamp() - dt.timestamp()) * 1000.
            if diff_dt < 0.:
                diff_dt = 0.
            packet_type=1
            if video_tag.codec_id == 7:
                avc_packet = await read_flv_avc_packet(reader)
                packet_type=avc_packet.packet_type
            if packet_type == 0:
                print(f'AVC sequence header; {str(video_tag)}; timestamp: {tag.timestamp}; diff(ms): [ts: {diff_ts}; dt: {int(diff_dt)}]')
            else:
                print(f'v_frame: {str(video_tag)}; timestamp: {tag.timestamp}; diff(ms): [ts: {diff_ts}; dt: {int(diff_dt)}]')
            timestamp[tag.type] = tag.timestamp
            dt = datetime.datetime.now()
            await read_video_flv_unit(reader, avc_dumper, tag, packet_type)
        except EOFError:
            break


async def read_cdn_header(reader, buffer):
    buffer += await read_bytes(reader, CDNHeader.size - len(buffer))
    if not buffer:
        raise EOFError
    return CDNHeader(buffer)


async def read_cdn_config(reader, cdn_header):
    ret=dict()
    for i in range(0, cdn_header.stream_count):
        buf = await read_bytes(reader, 5)
        if not buf:
            raise EOFError
        config=b''
        try:
            if buf[4] == 0x7b:
                stream_type,config_size=struct.unpack('<HH', buf[0:4])
                config=await read_bytes(reader, config_size-1)
                ret[stream_type]='{'+config.decode('utf-8')
            else:
                buf = buf + await read_bytes(reader, 5)
                stream_type,config_ts_and_size = struct.unpack('<HQ', buf)
                config=await read_bytes(reader, config_ts_and_size & 0xffff)
                ret[stream_type]=config.decode('utf-8')
        except UnicodeDecodeError as error:
            print(f'buf: {buf} config {config}')
            raise error
    return ret


async def read_cdn_stream_header(reader):
    buffer=await read_bytes(reader, CDNStreamHeader.size)
    if not buffer:
        raise EOFError
    return CDNStreamHeader(buffer)



def calc_timestamp_diff(prev_ts, next_ts):
    if next_ts > prev_ts:
        return next_ts-prev_ts
    return 0

def store_timestamp(store, stream_type, timestamp):
    if not stream_type in store:
        store[stream_type] = timestamp
    diff = calc_timestamp_diff(store[stream_type], timestamp)
    store[stream_type] = timestamp
    return diff


async def read_cdn(reader, buffer, avc_dumper, terminated_flag):
    cdn_header = await read_cdn_header(reader, buffer)
    print(str(cdn_header))
    cdn_config = await read_cdn_config(reader, cdn_header)
    print('archive streams:')
    for s_type,s_cfg in cdn_config.items():
        print(f'{s_type}. {s_cfg}')
    ts_timestamp=dict()
    abs_timestamp=dict()
    dt=datetime.datetime.now()
    while not terminated_flag.is_set():
        try:
            stream_header=await read_cdn_stream_header(reader)
            buf=await read_bytes(reader, stream_header.st_size)

            if stream_header.st_type != 1:
                diff_ts=store_timestamp(ts_timestamp, stream_header.st_type, stream_header.st_ts)
                diff_ats=store_timestamp(abs_timestamp, stream_header.st_type, stream_header.st_ats)
                print(str(stream_header), end='; ')
                print(f'diff(msec): [ts={diff_ts}; ats={diff_ats}]')
                continue

            diff_dt=calc_timestamp_diff(dt.timestamp(), datetime.datetime.now().timestamp()) * 1000.
            if (buf[0]&0x1f) == 1:
                diff_ts=store_timestamp(ts_timestamp, stream_header.st_type, stream_header.st_ts)
                diff_ats=store_timestamp(abs_timestamp, stream_header.st_type, stream_header.st_ats)
                print('video interframe', end=' ')
                print(str(stream_header), end='; ')
                print(f'diff(msec): [ts={diff_ts}; ats={diff_ats}; dt={int(diff_dt)}]')
                if avc_dumper:
                    avc_dumper.dump(buf)
                    avc_dumper.decode()
            elif (buf[0] & 0x1f) == 5:
                diff_ts=store_timestamp(ts_timestamp, stream_header.st_type, stream_header.st_ts)
                diff_ats=store_timestamp(abs_timestamp, stream_header.st_type, stream_header.st_ats)
                print('video keyframe', end='  ')
                print(str(stream_header), end= '; ')
                print(f'diff(msec): [ts={diff_ts}; ats={diff_ats}; dt={int(diff_dt)}]')
                if avc_dumper:
                    avc_dumper.dump(buf)
                    avc_dumper.decode()
            elif (buf[0]&0x1f) == 6:
                try:
                    sei=Sei(buf[1:11])
                    print(f'{str(sei)}; diff: {sei.get_diff(sei.timestamp)}')
                except:
                    print('unknown SEI:', end=' ')
                    for x in buf:
                        print(f'{hex(x)}', end=' ')
                    print('')
            elif (buf[0]&0x1f) == 7:
                if avc_dumper:
                    avc_dumper.dump(buf)
                print(Sps(buf[0], buf[1:]))
            elif (buf[0]&0x1f) == 8:
                if avc_dumper:
                    avc_dumper.dump(buf)
                print(Pps(buf[0], buf[1:]))
            else:
                #print(str(stream_header))
                pass
            dt = datetime.datetime.now()
        except EOFError:
            terminated_flag.set()
            break
        except KeyboardInterrupt:
            terminated_flag.set()
            break


async def print_http_headers(**argv):
    url = urllib.parse.urlsplit(argv.get('url'))
    if url.scheme == 'http':
        reader, writer = await asyncio.open_connection(url.hostname, url.port)
    elif url.scheme == 'https':
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        reader, writer = await asyncio.open_connection(host=url.hostname, port=url.port, ssl=ssl_context)
    else:
        print(f'invalid scheme {url.scheme}. http and https are only applied')
        return

    path = f"{url.path or '/'}"
    path += '?' + url.query if url.query else ''
    query = (
        f"GET {path} HTTP/1.0\r\n"
        f"Host: {url.hostname}\r\n"
        f"\r\n"
    )
    print(query)

    writer.write(query.encode('latin-1'))
    dump_avc=DumpAvc(path=path,
                     address=url.scheme+'://'+url.hostname+':'+str(url.port),
                     password=argv.get('password'),
                     audio=argv.get('audio'),
                     terminated=argv.get('terminated')) if argv.get('visual', False) else None
    try:
        is_flv,buf = await read_http_headers(reader)
        if is_flv:
            await read_flv(reader=reader,
                           dumper=dump_avc,
                           terminated=argv.get('terminated'))
        else:
            await read_cdn(reader, buf, dump_avc, argv.get('terminated'))
    except EOFError:
        pass
    writer.close()
    await writer.wait_closed()


if __name__ == '__main__':
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='flv[cdn] debugger')
    parser.add_argument('url', type=str, help='url to flv[cdn] source')
    parser.add_argument('--visual', action='store_true', help='show parsed frames')
    parser.add_argument('--audio', action='store_true', help='play audio samples')
    parser.add_argument('--password', type=str, help='password to decode path hash')
    args: argparse.Namespace = parser.parse_args()

    terminated=threading.Event()
    try:
        asyncio.run(print_http_headers(url=args.url,
                                       visual=args.visual,
                                       audio=args.audio,
                                       password=args.password,
                                       terminated=terminated))
    except KeyboardInterrupt:
        pass
    terminated.set()

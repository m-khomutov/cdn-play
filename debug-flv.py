import sys
import struct


class Header:
    def __init__(self, file):
        self._buffer = f.read(9)
        if chr(self._buffer[0]) != 'F' or chr(self._buffer[1]) != 'L' or chr(self._buffer[2]) != 'V' or self._buffer[3] != 1:
            raise SyntaxError('Invalid FLV header')
        print(f'signature: {chr(self._buffer[0])}{chr(self._buffer[1])}{chr(self._buffer[2])}')

    def __bytes__(self):
        return self._buffer


class TagSize:
    def __init__(self, file):
        self._buffer = f.read(4)
        if self._buffer is None:
            raise EOFError
        self.value = struct.unpack('>I', self._buffer)

    def __repr__(self):
        return f'{self.value}'

    def __bytes__(self):
        return self._buffer


class Sps:
    def __init__(self, file, nalu_type, sz):
        self.data = ''
        self._buffer=nalu_type.to_bytes()+file.read(sz)
        for x in self._buffer:
            self.data += hex(int(x)) + ' '

    def __repr__(self):
        return f'sps: {self.data}'

    def __bytes__(self):
        return self._buffer#[1:]


class Pps:
    def __init__(self, file, nalu_type, sz):
        self.data = ''
        self._buffer=nalu_type.to_bytes()+file.read(sz)
        for x in self._buffer:
            self.data += hex(int(x)) + ' '

    def __repr__(self):
        return f'pps: {self.data}'

    def __bytes__(self):
        return self._buffer#[1:]


class Sei:
    def __init__(self, src, sz):
        self.type = int(f.read(1)[0])
        self.size = int(f.read(1)[0])
        self.timestamp = struct.unpack('<Q', src.read(self.size))[0]
        src.read(sz-(2+self.size))

    def __repr__(self):
        return f'sei: type={self.type} ts={self.timestamp}'


class Tag:
    def __init__(self, file):
        self._data = f.read(11)
        self.type = int(self._data[0]) # 8, 9, 18
        self.nalu_type = 0
        self.packet_type = 0
        self.frame_mode = "sequence header"
        self.sps = None
        self.pps = None
        self.sei = None
        buffer = b'\x00' + self._data[1:4] # data size
        self.data_size = struct.unpack('>I', buffer)[0]
        buffer = self._data[4:7] # timestamp
        buffer = self._data[7:8] + buffer # timestamp extended
        self.timestamp = struct.unpack('>I', buffer)[0]
        if self.type == 9:
            buffer = f.read(5) # FrameType + CodecID + packetType + Composition Time
            self.packet_type = int(buffer[1])
            self._data = self._data+buffer
            if self.packet_type == 1:
                self.frame_mode = "nonIDR"
                off = self.data_size-10
                while off > 0:
                    buffer = f.read(5) # size avcC
                    nalu_size=struct.unpack('>I', buffer[0:4])[0]
                    self.nalu_type = int(buffer[4])
                    self._data = self._data+buffer
                    if (self.nalu_type & 0x1f) == 6:
                        self.sei = Sei(f, nalu_size-1)
                        off -= nalu_size+4
                    elif (self.nalu_type & 0x1f) == 7:
                        self.sps=Sps(file, self.nalu_type, nalu_size-1)
                        self._data = self._data+bytes(self.sps)
                        off -= nalu_size-1
                    elif (self.nalu_type & 0x1f) == 8:
                        self.pps=Pps(file, self.nalu_type, nalu_size-1)
                        self._data = self._data+bytes(self.pps)
                        off -= nalu_size+4
                    else:
                        off -= 5
                        self._data = self._data+f.read(nalu_size-1)
                        break
                if (self.nalu_type & 0x1f) == 5:
                    self.frame_mode = "IDR"
                return
            self._data = self._data+f.read(self.data_size-5)
        else:
            self._data = self._data+f.read(self.data_size)

    def __repr__(self):
        if self.type == 9:
            if self.packet_type == 0:
                return f'video; type: {self.type}; {self.frame_mode}; size: {self.data_size}; timestamp: {self.timestamp}'
            else:
                return f'video; type: {self.type}; nalu: {hex(self.nalu_type)} ({self.frame_mode}); size: {self.data_size}; timestamp: {self.timestamp}'
        return f'type: {self.type}; timestamp: {self.timestamp}'

    def __bytes__(self):
        return self._data



def dump_flv(filename, data, mode):
    if filename:
        with open(filename, mode) as dump_file:
            dump_file.write(data)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'run {sys.argv[0]} flvfile [dump]')
        sys.exit(1)
    dump_name = sys.argv[2] if len(sys.argv) == 3 else None
    with open(sys.argv[1], 'b+r') as f:
        hdr = Header(f)
        dump_flv(dump_name, b'\x00\x00\x00\x01', 'wb')
        #dump_flv(dump_name, bytes(hdr), 'wb')
        prev_timestamp = dict()
        prev_sei_timestamp = None
        while True:
            try:
                tag_size = TagSize(f)
                tag = Tag(f)
                #dump_flv(dump_name, bytes(tag_size), 'ab')
                #dump_flv(dump_name, bytes(tag), 'ab')
                if not tag.type in prev_timestamp:
                    prev_timestamp[tag.type] = tag.timestamp
                if tag.type != 9 or tag.packet_type != 1:
                    print(f'{str(tag)} diff: {tag.timestamp - prev_timestamp[tag.type]}')
                    prev_timestamp[tag.type] = tag.timestamp
                    continue
                if tag.sps:
                    print(f'{tag.sps}')
                    dump_flv(dump_name, bytes(tag.sps), 'ab')
                    dump_flv(dump_name, b'\x00\x00\x00\x01', 'ab')
                if tag.pps:
                    dump_flv(dump_name, bytes(tag.pps), 'ab')
                    dump_flv(dump_name, b'\x00\x00\x00\x01', 'ab')
                    print(f'{tag.pps}')
                if tag.sei:
                    if not prev_sei_timestamp:
                        prev_sei_timestamp = tag.sei.timestamp
                    print(f'{tag.sei} ts_diff: {tag.sei.timestamp - prev_sei_timestamp}')
                    prev_sei_timestamp = tag.sei.timestamp
                print(f'{str(tag)} diff: {tag.timestamp - prev_timestamp[tag.type]}')
                prev_timestamp[tag.type] = tag.timestamp
            except EOFError as e:
                print(f'end of file on {f.tell()}')
                break
            except IndexError as e:
                print(f'end of index on {f.tell()}')
                break
            except struct.error as e:
                print(f'end of index on {f.tell()}')
                break


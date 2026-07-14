import argparse
import asyncio
import random
import time
import urllib.parse


class DebugException(Exception):
    pass

async def read_idr(reader, size):
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
                raise DebugException(f'invalid symbol {hex(c[0])} in HTTP header')
            if int(c[0]) == 13 or int(c[0]) == 10:
                end_sequence += 1
            else:
                end_sequence = 0
        if end_sequence == 4:
            break
    print(ret.decode("utf-8"))
    return ret


async def make_idr_request(**argv):
    url = urllib.parse.urlsplit(argv.get('url'))
    if url.scheme != 'http':
        print(f'invalid scheme {url.scheme}. http is only applied')
        return

    reader, writer = await asyncio.open_connection(url.hostname, url.port)

    channels = argv.get('channels').split(' ')
    if not channels:
        raise DebugException('no channels to get IDR from')
    running = True
    while running:
        try:
            chan = random.choice(channels)
            path = f"/{chan}/idr"
            query = (
                f"GET {path} HTTP/1.0\r\n"
                f"Host: {url.hostname}\r\n"
                f'User-Agent: {argv.get("user_agent", "Chrome")}\r\n'
                f"\r\n"
            )
            print(query)
            writer.write(query.encode('latin-1'))
            try:
                result = 0
                buf = await read_http_headers(reader)
                headers = buf.decode('utf-8').split('\r\n')
                for h in headers:
                    if 'HTTP/1.' in h:
                        result = int(h.split(' ')[1])
                    if 'Content-Length: ' in h:
                        size = int(h.split(': ')[1])
                        filename = f'{chan}.264' if result == 200 else f'{chan}.error'
                        buf = await read_idr(reader=reader, size=size)
                        with open(filename, 'wb') as idr_file:
                            idr_file.write(buf)
                            print(f'saved: {filename} of {len(buf)} size\n')
                time.sleep(argv.get('timeout')/1000)
            except (EOFError, DebugException) as err:
                print(f'{err}')
                running = False
        except KeyboardInterrupt:
            running = False
    writer.close()
    await writer.wait_closed()


if __name__ == '__main__':
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='IDR requester')
    parser.add_argument('url', type=str, help='url to source')
    parser.add_argument('--channels', type=str, help='list of channels')
    parser.add_argument('--timeout', type=int, default=100, help='timeout msec.')
    args: argparse.Namespace = parser.parse_args()
    try:
        asyncio.run(make_idr_request(url=args.url,channels=args.channels,timeout=args.timeout))
    except KeyboardInterrupt:
        pass

"""
Client API for receiving images from a Filers server over the network (or locally).
To install::
    pip install tree_config ffpyplayer
To use, see `RemoteVideoPlayer` and the sample script at the end.
Source: https://gist.github.com/matham/1e057fc5556ec946900369a79b11df8f
"""

from itertools import accumulate
import traceback
import sys
import struct
from functools import partial
import socket
from queue import Queue, Empty
import select
from threading import Thread

from tree_config.utils import (
    yaml_loads as orig_yaml_loads,
    get_yaml,
    yaml_dumps as orig_yaml_dumps,
)
from ffpyplayer.pic import Image, SWScale


def yaml_loads(value):
    # somehow in older versions, b'yuv420p' got encoded to
    # '!!binary |\neXV2NDIwcA==\n' instead of '!!binary |\n eXV2NDIwcA==\n'.
    # So now it can't be parsed. Hence add the space
    if len(value) >= 12 and value.startswith("!!binary |\n") and value[11] != " ":
        value = value[:11] + " " + value[11:]
    return orig_yaml_loads(value, get_yaml_obj=get_yaml)


yaml_dumps = partial(orig_yaml_dumps, get_yaml_obj=get_yaml)


class EndConnection(Exception):
    """Raised when the socket connection is closed."""

    pass


connection_errors = (EndConnection, ConnectionAbortedError, ConnectionResetError)
"""Network connection exceptions we handle."""


class RemoteData:
    """Internal class that handles sending and receiving messages from a socket."""

    def send_msg(self, sock, msg, value):
        """Sends message to the server.
        :param sock: The socket
        :param msg: The message name string (e.g. image).
        :param value: The message value.
        :return:
        """
        if msg == "image":
            image, metadata = value
            bin_data = image.to_bytearray()
            data = yaml_dumps(
                (
                    "image",
                    (
                        list(map(len, bin_data)),
                        image.get_pixel_format(),
                        image.get_size(),
                        image.get_linesizes(),
                        metadata,
                    ),
                )
            )
            data = data.encode("utf8")
        else:
            data = yaml_dumps((msg, value))
            data = data.encode("utf8")
            bin_data = []

        sock.sendall(struct.pack(">II", len(data), sum(map(len, bin_data))))
        sock.sendall(data)
        for item in bin_data:
            sock.sendall(item)

    def decode_data(self, msg_buff, msg_len):
        """Decodes buffer data received from the network.
        :param msg_buff: The bytes data received so far.
        :param msg_len: The expected size of the message as tuple -
            The size of the message and any associated binary data.
        :return: A tuple of the message name and value, or (None, None) if
            we haven't read the full message.
        """
        n, bin_n = msg_len
        assert n + bin_n == len(msg_buff)
        data = msg_buff[:n].decode("utf8")
        msg, value = yaml_loads(data)

        if msg == "image":
            bin_data = msg_buff[n:]
            planes_sizes, pix_fmt, size, linesize, metadata = value
            starts = list(accumulate([0] + list(planes_sizes[:-1])))
            ends = accumulate(planes_sizes)
            planes = [bin_data[s:e] for s, e in zip(starts, ends)]

            value = planes, pix_fmt, size, linesize, metadata
        else:
            assert not bin_n
        return msg, value

    def read_msg(self, sock, msg_len, msg_buff):
        """Reads the message and decodes it once we read the full message.
        :param sock: The socket.
        :param msg_len: The tuple of the message length and associated data.
            If empty, the start of the next message will provide this
            information.
        :param msg_buff: The message buffer.
        :return: A 4-tuple of ``(msg_len, msg_buff, msg, value)``. Where
            ``msg_len, msg_buff`` are similar to the input, and
            ``(msg, value)`` is the message and its value if we read a full
            message, otherwise they are None.
        """
        # still reading msg size
        msg = value = None
        if not msg_len:
            assert 8 - len(msg_buff)
            data = sock.recv(8 - len(msg_buff))
            if not data:
                raise EndConnection("Remote client was closed")

            msg_buff += data
            if len(msg_buff) == 8:
                msg_len = struct.unpack(">II", msg_buff)
                msg_buff = b""
        else:
            total = sum(msg_len)
            assert total - len(msg_buff)
            data = sock.recv(total - len(msg_buff))
            if not data:
                raise EndConnection("Remote client was closed")

            msg_buff += data
            if len(msg_buff) == total:
                msg, value = self.decode_data(msg_buff, msg_len)

                msg_len = ()
                msg_buff = b""
        return msg_len, msg_buff, msg, value


class RemoteVideoPlayer(RemoteData):
    """A player that is a network client that plays images received via a
    :class:`RemoteData` over the network.
    The typical process is:
    #. Start playing a camera in Filers.
    #. Set up the recorder to the network server,
       #. Provide a IP address it'll serve the images, e.g. "localhost" if it's only served on the local computer.
       #. Provide a port, a number above 5000 to serve the images on.
       #. Start the server.
       #. Start recording images to the network. If there are no clients connected, no images are actually sent. You can
          also start recording after a client has connected and requested to play.
    #. On the client side,
       #. Create a `TestRemoteVideoPlayer` with the given IP and port.
       #. Call `start_listener` to connect to the server so we can communicate with it. You must now constantly call
          `process_in_main_thread` to handle incoming messages from the server.
       #. When you're ready to handle images, call `play`. If the server is already recording it'll immediately start
          sending messages. Otherwise, it'll start sending messages when you click record in Filers. Either way it'll
          first send a `process_recording_state` message so you know to expect images.
    #. On the client side you can hit `stop` and the server will stop sending images. Or on the server side you can
       stop recording and then the server will send a `process_recording_state` to let you know to stop expecting
       images until the next time you start recording in Filers.
       You can circle back to the last step to start again.
    #. Client calls `stop` and then `stop_listener` to disconnect from the server.
    If there are any exceptions from the server or the internal thread, `process_exception` is called (via
    `process_in_main_thread`). Similarly, `process_recording_state` is called (also via `process_in_main_thread`)
    when the server changes it's recording state. `process_image` is called (also via `process_in_main_thread`) for
    each image we get from the server.
    """

    server: str = ""
    """The server address that broadcasts the data.
    """

    port: int = 0
    """The server port that broadcasts the data.
    """

    timeout: float = 0.01
    """How long to wait before timing out when reading data before checking the
    queue for other requests.
    """

    _client_active: bool = False

    _listener_thread: Thread | None = None

    _to_main_thread_queue: Queue | None = None

    _from_main_thread_queue: Queue | None = None

    def __init__(self, server: str, port: int, timeout: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.server = server
        self.port = port
        self.timeout = timeout

    def process_exception(self, e, exec_info, from_thread: bool = False):
        """Called when we get an error from the server or from the internal thread.
        :param e: The error message string.
        :param exec_info: The error stack.
        :param from_thread: Bool, which is True if the error comes from our own 2nd thread or False when it's from
            the server.
        """
        raise NotImplementedError

    def process_recording_state(self, state: bool):
        """Called when the recording state of the server changes.
        :param state: If True, it means the server started recording video to the network and we should expect images.
            We get this message from the server after we call `play` when the server starts sending messages to us.
            If False, the server stopped recording images for us and we should stop expecting images until
            we get another start message.
        """
        raise NotImplementedError

    def process_image(self, image: Image, metadata: dict):
        """Called for each image we receive from the server.
        :param image: An `ffpyplayer.pic.Image` instance.
        :param metadata: A dict with metadata about the image. Such as the timestamp from the server when the image was
            taken. At a minimum it'll have a `"t"` key that is the server timestamp of the image, in seconds.
        :return:
        """
        raise NotImplementedError

    def _listener_run(self, _from_main_thread_queue, _to_main_thread_queue):
        """Client method, that is executed in the internal client thread."""
        timeout = self.timeout

        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)

        # Connect the socket to the port where the server is listening
        server_address = (self.server, self.port)
        print("RemoteVideoPlayer: connecting to {} port {}".format(*server_address))

        msg_len, msg_buff = (), b""

        try:
            sock.connect(server_address)
            done = False

            while not done:
                r, _, _ = select.select([sock], [], [], timeout)
                if r:
                    msg_len, msg_buff, msg, value = self.read_msg(
                        sock, msg_len, msg_buff
                    )
                    if msg is not None:
                        _to_main_thread_queue.put((msg, value))

                try:
                    while True:
                        msg, value = _from_main_thread_queue.get_nowait()
                        if msg == "eof":
                            done = True
                            break
                        else:
                            self.send_msg(sock, msg, value)
                except Empty:
                    pass
        except Exception as e:
            exc_info = "".join(traceback.format_exception(*sys.exc_info()))
            _to_main_thread_queue.put(("exception_exit", (str(e), exc_info)))
        finally:
            print("RemoteVideoPlayer: closing socket")
            sock.close()

    def _send_message_to_server(self, key, value):
        """Sends the message to the server over the network.
        :param msg: The message name string.
        :param value: The message value.
        """
        if self._from_main_thread_queue is None:
            return
        self._from_main_thread_queue.put((key, value))

    def process_in_main_thread(self):
        """This must be called frequently in the main thread to process any incoming messages from the server.
        This calls `process_exception`, `process_recording_state`, or `process_image` in response to incoming
        server messages.
        """
        while self._to_main_thread_queue is not None:
            try:
                msg, value = self._to_main_thread_queue.get(block=False)

                if msg == "exception":
                    e, exec_info = value
                    self.process_exception(e, exec_info)
                elif msg == "exception_exit":
                    e, exec_info = value
                    self.process_exception(e, exec_info, from_thread=True)
                elif msg == "started_recording":
                    self.process_recording_state(True)
                elif msg == "stopped_recording":
                    self.process_recording_state(False)
                elif msg == "stopped_playing":
                    pass
                elif msg == "image":
                    plane_buffers, pix_fmt, size, linesize, metadata = value
                    sws = SWScale(*size, pix_fmt, ofmt=pix_fmt)
                    img = Image(
                        plane_buffers=plane_buffers,
                        pix_fmt=pix_fmt,
                        size=size,
                        linesize=linesize,
                    )
                    self.process_image(sws.scale(img), metadata)
                else:
                    print("Got unknown RemoteVideoPlayer message", msg, value)
            except Empty:
                break

    def start_listener(self):
        """Call this to start the internal thread that connects to the server and listens to server messages.
        The server should be already running when this is called. Otherwise, you'll get a timeout exception
        through `process_exception`. Once called, you should call `process_in_main_thread` repeatedly to
        handle incoming server messages. You can also then call `play` or `stop`.
        """
        # we make sure client is connected and request metadata. If the player
        # is not playing on the server, we cannot ask it to play so we wait
        # until it sends us the metadata. If it was already playing, it sends
        # it again, otherwise, it'll send it to us when it starts playing

        if self._listener_thread is not None:
            return

        self._client_active = True
        _from_main_thread_queue = self._from_main_thread_queue = Queue()
        _to_main_thread_queue = self._to_main_thread_queue = Queue()
        thread = self._listener_thread = Thread(
            target=self._listener_run,
            args=(_from_main_thread_queue, _to_main_thread_queue),
        )
        thread.start()

    def stop_listener(self, join=True):
        """Call to stop our internal thread and to stop listening to server messages.
        :param join: Whether to wait until the internal thread has exited.
        """
        self.stop()

        if self._listener_thread is None:
            return

        self._from_main_thread_queue.put(("eof", None))
        if join:
            self._listener_thread.join()

        self._listener_thread = self._to_main_thread_queue = (
            self._from_main_thread_queue
        ) = None
        self._client_active = False

    def play(self):
        """After calling `start_listener` and we are connected to the server, we can call this to let the server
        know that we want the server to send us images from the camera.
        On the server side, the server will only send us images after play was called.
        """
        self._send_message_to_server("started_playing", None)

    def stop(self):
        """Like `play`, but to let the server know that we stopped playing server images and for the server to stop
        sending us images. The server won't send us any further images until `play` is called again.
        """
        self._send_message_to_server("stopped_playing", None)


if __name__ == "__main__":
    import time

    class TestRemoteVideoPlayer(RemoteVideoPlayer):
        got_exception = False

        def process_exception(self, e, exec_info, from_thread: bool = False):
            print(f"Got exception: '{e}'")
            self.got_exception = True

        def process_recording_state(self, state: bool):
            if state:
                print("Player started recording")
            else:
                print("Player stopped recording")

        def process_image(self, image: Image, metadata: dict):
            print(f"Got image: {image}, {metadata}")

    player = TestRemoteVideoPlayer(server="localhost", port=5002)
    player.start_listener()
    player.play()

    ts = time.perf_counter()
    while time.perf_counter() - ts < 10 and not player.got_exception:
        player.process_in_main_thread()
        time.sleep(0.05)

    player.stop()
    player.stop_listener()

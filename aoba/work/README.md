# socket 通信のテンプレ

```bash
$ port=50000
$ python aoba/work/build_socket_server.py --port $port  # 対話ホストの起動
$ python aoba/work/client.py --port $port              # 対話システムの起動
```
"""允许 ``python -m betatrend`` 直接进入 CLI。

``__main__`` 只做入口转发：真正的子命令解析、回测、paper、探活都在
``betatrend.cli.main`` 里。用 ``raise SystemExit`` 把 CLI 的返回码交给操作系统。
"""

from betatrend.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

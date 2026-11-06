"""`gateway` command: serve, manage keys, run a mock upstream."""

from __future__ import annotations

import argparse
import logging

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gateway", description="OpenAI-compatible LLM gateway")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the gateway")
    s.add_argument("--config", default="gateway.toml")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--access-log", action="store_true")

    k = sub.add_parser("keys", help="manage API keys")
    k.add_argument("--config", default="gateway.toml")
    ksub = k.add_subparsers(dest="keys_cmd", required=True)
    kc = ksub.add_parser("create")
    kc.add_argument("name")
    kc.add_argument("--rpm", type=int)
    kc.add_argument("--tpm", type=int)
    ksub.add_parser("list")

    m = sub.add_parser("mock-upstream", help="run a fake OpenAI-compatible upstream for benchmarks")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=9001)
    m.add_argument("--latency-ms", type=float, default=0.0)
    m.add_argument("--fail-rate", type=float, default=0.0)
    m.add_argument("--token-delay-ms", type=float, default=0.0)
    m.add_argument("--seed", type=int)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "serve":
        import uvicorn

        from .app import create_app

        uvicorn.run(create_app(load_config(args.config)), host=args.host, port=args.port,
                    access_log=args.access_log, log_level="info")
    elif args.cmd == "keys":
        from .store import Store

        cfg = load_config(args.config)
        store = Store(cfg.db_path)
        if args.keys_cmd == "create":
            key, rec = store.create_key(args.name, args.rpm or cfg.limits.default_rpm,
                                        args.tpm or cfg.limits.default_tpm)
            print(f"created key #{rec.id} {rec.name!r} (rpm={rec.rpm}, tpm={rec.tpm})")
            print(key)
            print("store it now: only its hash is kept")
        else:
            for rec in store.list_keys():
                print(f"#{rec.id:<4} {rec.prefix}...  {rec.name:<20} rpm={rec.rpm:<6} tpm={rec.tpm:<8}"
                      f"{' disabled' if rec.disabled else ''}")
        store.close()
    elif args.cmd == "mock-upstream":
        import uvicorn

        from .mock_upstream import create_mock_app

        uvicorn.run(create_mock_app(args.latency_ms, args.fail_rate, args.token_delay_ms, args.seed),
                    host=args.host, port=args.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()

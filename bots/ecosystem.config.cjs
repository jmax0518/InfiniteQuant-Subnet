/**
 * PM2 process file for SN89 local tools.
 *
 *   cd InfiniteQuant-Subnet/bots
 *   pm2 start ecosystem.config.cjs
 *   pm2 save
 *
 * Dashboard: http://0.0.0.0:8899
 * Miner serve (GOLD/sn89): pm2 start ecosystem.config.cjs --only sn89-miner-serve
 * HF auto-bot (mechanism 1, dry-run by default): pm2 start ecosystem.config.cjs --only sn89-hf-auto
 *   Set SN89_HF_AUTO_LIVE=1 before `pm2 start`/`pm2 restart` to actually submit_hf().
 *   Pause/resume without restarting from the dashboard's HF Auto panel.
 *   Runs on its own dedicated hotkey (JMax-1/sn89-1, UID 199) by default —
 *   override with SN89_HF_WALLET_NAME / SN89_HF_WALLET_HOTKEY / SN89_HF_HOTKEY_SS58.
 */
const path = require("path");
const fs = require("fs");

const ROOT = path.resolve(__dirname, "..");
const BOTS = __dirname;
const MVTRX = process.env.SN89_MINER_ROOT || "/root/MVTRX_08_05/InfiniteQuant-Subnet";
const PY =
  process.env.SN89_PYTHON ||
  (fs.existsSync(path.join(ROOT, ".venv", "bin", "python"))
    ? path.join(ROOT, ".venv", "bin", "python")
    : "python3");

// hf_auto_bot.py imports neurons.miner in-process (submit_hf latency), which
// needs miner deps like `timelock` that only live in the working miner venv —
// same one ai_finance_bot.py's CLI-fallback subprocess and start-miner-serve.sh
// already trust for real chain calls. Bare python3 doesn't have these.
const HF_PY =
  process.env.SN89_HF_PYTHON ||
  (fs.existsSync(path.join(MVTRX, ".venv", "bin", "python"))
    ? path.join(MVTRX, ".venv", "bin", "python")
    : PY);

function loadSn89Env() {
  const envPath = "/root/.sn89/env";
  const out = {};
  if (!fs.existsSync(envPath)) return out;
  for (const line of fs.readFileSync(envPath, "utf8").split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#")) continue;
    const body = t.startsWith("export ") ? t.slice(7) : t;
    const i = body.indexOf("=");
    if (i < 1) continue;
    let k = body.slice(0, i).trim();
    let v = body.slice(i + 1).trim();
    if (
      (v.startsWith('"') && v.endsWith('"')) ||
      (v.startsWith("'") && v.endsWith("'"))
    ) {
      v = v.slice(1, -1);
    }
    out[k] = v;
  }
  return out;
}

const sn89Env = loadSn89Env();

// Force-clear any proxy vars inherited from whatever shell happens to run
// `pm2 start`/`restart` (e.g. a dev sandbox's HTTP_PROXY/HTTPS_PROXY) — pm2
// bakes the LAUNCHING shell's env into the daemonized process and keeps it
// across restarts, so a proxy picked up once here silently persists and
// breaks these processes' real outbound calls (partner.infinitequant.app,
// Binance, etc.) even long after that shell is gone. Spread this LAST in
// every app's env so it always wins over anything inherited.
const NO_PROXY_ENV = {
  HTTP_PROXY: "", HTTPS_PROXY: "", http_proxy: "", https_proxy: "",
  ALL_PROXY: "", all_proxy: "", SOCKS_PROXY: "", socks_proxy: "",
  SOCKS5_PROXY: "", socks5_proxy: "",
};

module.exports = {
  apps: [
    {
      name: "sn89-dashboard",
      cwd: ROOT,
      script: path.join(BOTS, "dashboard.py"),
      interpreter: PY,
      args: "--host 0.0.0.0 --port 8899",
      autorestart: true,
      max_restarts: 50,
      min_uptime: "5s",
      restart_delay: 2000,
      watch: false,
      env: {
        PYTHONUNBUFFERED: "1",
        WALLET_NAME: process.env.WALLET_NAME || "GOLD",
        // Own SN89 miner hotkey (do NOT use friend hotkey sn89)
        WALLET_HOTKEY: process.env.WALLET_HOTKEY || "iq89",
        SN89_HOTKEY_SS58:
          process.env.SN89_HOTKEY_SS58 ||
          "5Co94YcY8EDTAHcgFs5sB4dW4R999urfMDExXhAJxysk2gU8",
        SN89_MINER_ROOT: MVTRX,
        SN89_SUBMIT_MODE: process.env.SN89_SUBMIT_MODE || "auto",
        ...sn89Env,
        ...NO_PROXY_ENV,
      },
      error_file: path.join(BOTS, "logs", "dashboard.err.log"),
      out_file: path.join(BOTS, "logs", "dashboard.out.log"),
      merge_logs: true,
      time: true,
    },
    {
      name: "sn89-miner-serve",
      cwd: BOTS,
      script: path.join(BOTS, "start-miner-serve.sh"),
      interpreter: "bash",
      autorestart: true,
      max_restarts: 50,
      min_uptime: "5s",
      restart_delay: 3000,
      watch: false,
      env: {
        PYTHONUNBUFFERED: "1",
        WALLET_NAME: process.env.WALLET_NAME || "GOLD",
        WALLET_HOTKEY: process.env.WALLET_HOTKEY || "iq89",
        SN89_HOTKEY_SS58:
          process.env.SN89_HOTKEY_SS58 ||
          "5Co94YcY8EDTAHcgFs5sB4dW4R999urfMDExXhAJxysk2gU8",
        SN89_MINER_ROOT: MVTRX,
        SN89_SERVE_HOST: "127.0.0.1",
        SN89_SERVE_PORT: "8089",
        ...sn89Env,
        ...NO_PROXY_ENV,
      },
      error_file: path.join(BOTS, "logs", "miner-serve.err.log"),
      out_file: path.join(BOTS, "logs", "miner-serve.out.log"),
      merge_logs: true,
      time: true,
      autostart: true,
    },
    {
      name: "sn89-hf-auto",
      cwd: ROOT,
      script: path.join(BOTS, "hf_auto_bot.py"),
      interpreter: HF_PY,
      args: process.env.SN89_HF_AUTO_LIVE === "1" ? "--live" : "--dry-run",
      autorestart: true,
      max_restarts: 50,
      min_uptime: "5s",
      restart_delay: 3000,
      watch: false,
      env: {
        PYTHONUNBUFFERED: "1",
        // Dedicated HF-only identity (UID 199) — separate from the GOLD/iq89
        // LF hotkey so the two mechanisms never compete for the same rate
        // limits or trip the cross-mechanism lock against each other.
        WALLET_NAME: process.env.SN89_HF_WALLET_NAME || "JMax-1",
        WALLET_HOTKEY: process.env.SN89_HF_WALLET_HOTKEY || "sn89-1",
        SN89_HOTKEY_SS58:
          process.env.SN89_HF_HOTKEY_SS58 ||
          "5FPDM62bALVM2CMx5Cc84WjaKxVTqkBhoRxGe7npBe1DKxTd",
        // Explicit (not left to fall back on WALLET_HOTKEY) so the dashboard,
        // a SEPARATE process with its own unrelated WALLET_HOTKEY=GOLD/iq89,
        // can name this exact instance's state/enable files via ?miner=sn89-1.
        SN89_HF_AUTO_TAG: "sn89-1",
        ...sn89Env,
        ...NO_PROXY_ENV,
      },
      error_file: path.join(BOTS, "logs", "hf-auto.err.log"),
      out_file: path.join(BOTS, "logs", "hf-auto.out.log"),
      merge_logs: true,
      time: true,
    },
    {
      // Second HF miner slot (UID 202, JMax-1/sn89-2, registered 2026-08-31).
      // Same strategy/config as sn89-hf-auto (both read the same bots/.env
      // and ~/.sn89/hf_auto_bot.env) — just a second identity/submission
      // quota. Toggle SN89_HF_AUTO2_LIVE=1 independently of the first bot's
      // SN89_HF_AUTO_LIVE.
      name: "sn89-hf-auto-2",
      cwd: ROOT,
      script: path.join(BOTS, "hf_auto_bot.py"),
      interpreter: HF_PY,
      args: process.env.SN89_HF_AUTO2_LIVE === "1" ? "--live" : "--dry-run",
      autorestart: true,
      max_restarts: 50,
      min_uptime: "5s",
      restart_delay: 3000,
      watch: false,
      env: {
        PYTHONUNBUFFERED: "1",
        WALLET_NAME: process.env.SN89_HF_WALLET_NAME || "JMax-1",
        WALLET_HOTKEY: "sn89-2",
        SN89_HOTKEY_SS58:
          process.env.SN89_HF2_HOTKEY_SS58 ||
          "5ECaqgexkqF6E2XoT2LAybwayQKf34aMPNzfmaKvr8WV3pYz",
        SN89_HF_AUTO_TAG: "sn89-2",
        ...sn89Env,
        ...NO_PROXY_ENV,
      },
      error_file: path.join(BOTS, "logs", "hf-auto-2.err.log"),
      out_file: path.join(BOTS, "logs", "hf-auto-2.out.log"),
      merge_logs: true,
      time: true,
    },
    {
      name: "sn89-desk-tunnel",
      cwd: "/root/.sn89",
      script: "/root/.sn89/run-desk-tunnel.sh",
      interpreter: "bash",
      autorestart: true,
      max_restarts: 100,
      min_uptime: "10s",
      restart_delay: 5000,
      watch: false,
      error_file: "/root/.sn89/cloudflared.err.log",
      out_file: "/root/.sn89/cloudflared.out.log",
      merge_logs: true,
      time: true,
    },
  ],
};

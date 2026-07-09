export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=http://127.0.0.1:17890
export all_proxy=socks5://127.0.0.1:17890
export ALL_PROXY=socks5://127.0.0.1:17890
curl -I https://github.com
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
node -v
npm -v
# npm 走代理
npm config set proxy http://127.0.0.1:17890
npm config set https-proxy http://127.0.0.1:17890

# 安装 Codex CLI
npm install -g @openai/codex

# 检查
which codex
codex --version

codex \
  --sandbox danger-full-access \
  --ask-for-approval never

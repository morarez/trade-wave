#!/usr/bin/env node
const { spawn } = require('child_process');
const fs = require('fs');
const net = require('net');
const path = require('path');

const rootDir = path.resolve(__dirname, '..');

function resolvePythonCommand() {
  const candidates = [];

  if (process.platform === 'win32') {
    candidates.push(path.join(rootDir, 'venv', 'Scripts', 'python.exe'));
    candidates.push(path.join(rootDir, '.venv', 'Scripts', 'python.exe'));
  } else {
    candidates.push(path.join(rootDir, 'venv', 'bin', 'python'));
    candidates.push(path.join(rootDir, '.venv', 'bin', 'python'));
  }

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return process.env.PYTHON || (process.platform === 'win32' ? 'python.exe' : 'python3');
}

const backendCommand = resolvePythonCommand();
const frontendCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const frontendArgs = ['--prefix', 'frontend', 'run', 'dev'];

const children = [];
let shuttingDown = false;

function cleanup(signal = 'SIGTERM') {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) {
      child.kill(signal);
    }
  }
}

function findFreePort(startPort = 5000, endPort = 5010) {
  return new Promise((resolve, reject) => {
    const tryPort = (port) => {
      const server = net.createServer();
      server.once('error', (error) => {
        if (error.code === 'EADDRINUSE' && port < endPort) {
          tryPort(port + 1);
        } else {
          reject(error);
        }
      });
      server.once('listening', () => {
        server.close(() => resolve(port));
      });
      server.listen(port, '127.0.0.1');
    };

    tryPort(startPort);
  });
}

function waitForBackend(port, timeoutMs = 60000) {
  const startedAt = Date.now();
  return new Promise((resolve, reject) => {
    const tryConnect = () => {
      const socket = net.createConnection({ host: '127.0.0.1', port });
      socket.on('connect', () => {
        socket.end();
        resolve();
      });
      socket.on('error', () => {
        if (Date.now() - startedAt >= timeoutMs) {
          reject(new Error(`Backend did not become ready on port ${port} in time.`));
          return;
        }
        setTimeout(tryConnect, 500);
      });
    };

    tryConnect();
  });
}

function startProcess(label, command, args, cwd, extraEnv = {}) {
  const child = spawn(command, args, {
    cwd,
    stdio: 'inherit',
    shell: false,
    env: { ...process.env, ...extraEnv, FORCE_COLOR: '1' }
  });

  children.push(child);

  child.on('exit', (code, signal) => {
    if (label === 'Frontend' && code === 0 && !signal) {
      console.log(`\n${label} exited cleanly; continuing to run the backend.`);
      return;
    }

    if (signal) {
      console.log(`\n${label} stopped with signal ${signal}.`);
    } else {
      console.log(`\n${label} exited with code ${code}.`);
    }

    cleanup(signal || 'SIGTERM');
    process.exit(code === null ? 1 : code);
  });

  child.on('error', (error) => {
    console.error(`Failed to start ${label}:`, error.message);
    cleanup('SIGTERM');
    process.exit(1);
  });
}

process.on('SIGINT', () => {
  cleanup('SIGINT');
  process.exit(0);
});

process.on('SIGTERM', () => {
  cleanup('SIGTERM');
  process.exit(0);
});

async function main() {
  const backendPort = await findFreePort();

  console.log('Starting backend and frontend...');
  console.log(`Backend will run on http://127.0.0.1:${backendPort}`);

  startProcess(
    'Backend',
    backendCommand,
    ['main.py', 'serve', '--host', '127.0.0.1', '--port', String(backendPort)],
    rootDir
  );

  try {
    await waitForBackend(backendPort);
    console.log('Backend is ready. Starting frontend...');
    startProcess('Frontend', frontendCommand, frontendArgs, rootDir, {
      VITE_API_PORT: String(backendPort)
    });
  } catch (error) {
    console.error(error.message);
    cleanup('SIGTERM');
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('Failed to start development servers:', error.message);
  process.exit(1);
});

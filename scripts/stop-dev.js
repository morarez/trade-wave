#!/usr/bin/env node
const { spawnSync } = require('child_process');
const os = require('os');
const path = require('path');

function run(command, args) {
  const result = spawnSync(command, args, { encoding: 'utf8' });
  if (result.error) {
    throw result.error;
  }
  return result;
}

function killProcessByPort(port) {
  const commands = {
    darwin: ['lsof', ['-ti', `tcp:${port}`]],
    linux: ['lsof', ['-ti', `tcp:${port}`]],
    win32: ['netstat', ['-ano']]
  };

  const [command, args] = commands[process.platform] || commands.darwin;
  const result = run(command, args);

  if (result.status !== 0) return;

  const pids = result.stdout
    .split(/\s+/)
    .map((value) => value.trim())
    .filter(Boolean);

  for (const pid of pids) {
    if (process.platform === 'win32') {
      const match = result.stdout.match(new RegExp(`LISTENING\\s+${pid}`));
      if (match) {
        run('taskkill', ['/PID', pid, '/F']);
      }
    } else {
      run('kill', ['-9', pid]);
    }
  }
}

function killKnownTradeWaveProcesses() {
  killProcessByPort(5000);
  killProcessByPort(5001);
  killProcessByPort(5002);
  killProcessByPort(5003);
  killProcessByPort(5004);
  killProcessByPort(5005);

  const result = run('ps', ['aux']);
  const lines = result.stdout.split('\n') || [];
  const targets = ['node scripts/dev.js', 'vite', 'main.py serve', 'python main.py serve', 'python3 main.py serve'];

  for (const line of lines) {
    if (targets.some((target) => line.includes(target))) {
      const match = line.match(/\s+(\d+)\s+/);
      if (match) {
        const pid = match[1];
        run('kill', ['-9', pid]);
      }
    }
  }
}

killKnownTradeWaveProcesses();
console.log('Stopped Trade Wave dev processes if they were running.');

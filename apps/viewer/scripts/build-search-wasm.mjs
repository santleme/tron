#!/usr/bin/env node
import { mkdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const source = resolve(root, 'apps/viewer/search/territory.rs');
const output = resolve(root, 'apps/viewer/src/generated/fixed_depth.wasm');
await mkdir(dirname(output), { recursive: true });

const result = spawnSync('rustc', [
  '--crate-type', 'cdylib', '--target', 'wasm32-unknown-unknown',
  '--edition=2021', '--check-cfg=cfg(circuit_parity)', '-C', 'opt-level=z', '-C', 'lto', '-C', 'codegen-units=1',
  '-C', 'strip=symbols', source, '-o', output,
], { cwd: root, encoding: 'utf8' });
if (result.status !== 0) {
  process.stderr.write(result.stderr || result.stdout || 'rustc failed\n');
  process.exit(result.status ?? 1);
}
process.stdout.write(`Built fixed-depth planner WASM: ${output}\n`);

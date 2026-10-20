import { replayActions, verifyObjectDigest } from '../engine/replay.ts';
import { createSeededState } from '../engine/engine.ts';
import type { ArenaState, JointAction } from '../engine/types.ts';

export type CheckpointMetadata = {
  format: string; checkpoint_id: string; step: number; observation_schema: string; action_schema: string;
  model_architecture: { name: string; version: string; config: { width: number; height: number; hidden_size: number } };
  onnx: { file: string; sha256: string; opset: number; runtime: string };
  pytorch: { file: string; sha256: string; format: string };
  artifacts: { format: 'onnx'|'pytorch'; file: string; sha256: string }[];
  metadata_sha256: string;
};
export type ReplayArtifact = {
  format: string; run_id: string; match_id: string; engine_version: string; observation_schema: string; action_schema: string;
  master_seed: string; arena_index: number; derived_streams: {label:string;initstate:string;initseq:string}[];
  agents: {agent_id:string;checkpoint_id:string}[];
  initial_state: {width:number;height:number;positions:{x:number;y:number}[];headings:string[];trails:{x:number;y:number}[][]};
  ticks: {tick:number;joint_actions:JointAction[];collision_causes:string[][];terminal_causes:string[]}[];
  result: 'agent_0_win'|'agent_1_win'|'draw'; tick_count:number; replay_sha256:string;
};
export function canonicalJson(value: any): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${canonicalJson(value[k])}`).join(',')}}`;
  return JSON.stringify(value);
}
function hasOnlyKeys(value:unknown, keys:readonly string[]):boolean {
  return Boolean(value&&typeof value==='object'&&!Array.isArray(value)&&Object.keys(value).every(key=>keys.includes(key))&&keys.every(key=>Object.hasOwn(value,key)));
}
export async function validateCheckpointMetadata(value: CheckpointMetadata): Promise<void> {
  if(!hasOnlyKeys(value,['format','checkpoint_id','step','model_architecture','observation_schema','action_schema','onnx','pytorch','artifacts','metadata_sha256']))throw new Error('Checkpoint metadata has missing or unknown fields');
  if (value.format !== 'lc-checkpoint/1' || value.observation_schema !== 'lc-observation/1' || value.action_schema !== 'lc-action/1') throw new Error('Unsupported checkpoint metadata or schemas');
  if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value.checkpoint_id ?? '') || !Number.isSafeInteger(value.step) || value.step < 0) throw new Error('Checkpoint metadata has an invalid ID or step');
  if(!hasOnlyKeys(value.model_architecture,['name','version','config'])||!hasOnlyKeys(value.model_architecture?.config,['width','height','hidden_size']))throw new Error('Checkpoint model architecture config is incomplete or contains unknown fields');
  if (value.model_architecture?.name !== 'mlp-hwc-flatten' || value.model_architecture?.version !== '1') throw new Error('Unsupported checkpoint model architecture');
  const config=value.model_architecture.config;
  if (!config || !Number.isSafeInteger(config.width) || config.width < 2 || !Number.isSafeInteger(config.height) || config.height < 2 || !Number.isSafeInteger(config.hidden_size) || config.hidden_size < 1) throw new Error('Checkpoint architecture config must include valid width, height, and hidden_size');
  if (!hasOnlyKeys(value.onnx,['file','sha256','opset','runtime'])||value.onnx?.runtime !== 'onnxruntime-web') throw new Error('Checkpoint ONNX details are incomplete or runtime is unsupported');
  if (value.onnx?.opset !== 17) throw new Error(`Unsupported ONNX opset ${value.onnx?.opset}; expected 17`);
  const validHash=(hash:unknown):hash is string=>typeof hash==='string'&&/^[a-f0-9]{64}$/.test(hash);
  const safePath=(path:unknown):path is string=>typeof path==='string'&&path.length>0&&!path.startsWith('/')&&!path.includes('\\')&&!path.includes('\0')&&!path.split('/').some(part=>part==='..'||part==='.'||part==='');
  if (!safePath(value.onnx?.file) || !value.onnx.file.endsWith('.onnx') || !validHash(value.onnx?.sha256)) throw new Error('Checkpoint metadata has an invalid ONNX file path or SHA-256');
  if (!hasOnlyKeys(value.pytorch,['file','sha256','format'])||!safePath(value.pytorch.file) || !value.pytorch.file.endsWith('.pt') || !validHash(value.pytorch.sha256) || typeof value.pytorch.format!=='string' || value.pytorch.format.length===0) throw new Error('Checkpoint metadata has invalid PyTorch artifact details');
  if (!Array.isArray(value.artifacts) || value.artifacts.length!==2) throw new Error('Checkpoint metadata must list its ONNX and PyTorch artifacts');
  if(value.artifacts.some(artifact=>!hasOnlyKeys(artifact,['format','file','sha256'])))throw new Error('Checkpoint artifact entries are malformed');
  const onnxArtifact=value.artifacts.find(artifact=>artifact.format==='onnx'), pytorchArtifact=value.artifacts.find(artifact=>artifact.format==='pytorch');
  if (!onnxArtifact || onnxArtifact.file!==value.onnx.file || onnxArtifact.sha256!==value.onnx.sha256 || !safePath(onnxArtifact.file) || !validHash(onnxArtifact.sha256)) throw new Error('ONNX artifact list does not match the ONNX checkpoint entry');
  if (!pytorchArtifact || pytorchArtifact.file!==value.pytorch.file || pytorchArtifact.sha256!==value.pytorch.sha256 || !safePath(pytorchArtifact.file) || !validHash(pytorchArtifact.sha256)) throw new Error('PyTorch artifact list does not match the PyTorch checkpoint entry');
  if (!(await verifyObjectDigest(value as any,'metadata_sha256'))) throw new Error('Metadata digest is invalid');
}
export function assertCheckpointMetadataPresent(value:CheckpointMetadata|null):asserts value is CheckpointMetadata {
  if(!value)throw new Error('Load and verify checkpoint metadata before its ONNX model');
}

export type ModelValueMetadata={name:string;isTensor:boolean;type?:string;shape?:readonly (number|string)[]};
export function assertModelMatchesMetadata(metadata:CheckpointMetadata, fileName:string, modelSha256:string, inputNames:readonly string[], outputNames:readonly string[], inputMetadata:readonly ModelValueMetadata[], outputMetadata:readonly ModelValueMetadata[]):void {
  if(metadata.onnx.file.split('/').at(-1)!==fileName)throw new Error('Loaded ONNX filename does not match checkpoint metadata');
  if(metadata.onnx.sha256!==modelSha256)throw new Error('Checkpoint SHA-256 does not match its metadata');
  if(inputNames.length!==1||inputNames[0]!=='observation'||!outputNames.includes('logits'))throw new Error('ONNX model inputs/outputs do not match the checkpoint contract');
  const input=inputMetadata.find(value=>value.name==='observation'), logits=outputMetadata.find(value=>value.name==='logits'), dims=metadata.model_architecture.config;
  if(!input?.isTensor||input.type!=='float32'||input.shape?.length!==4||input.shape[3]!==12||(typeof input.shape[1]==='number'&&input.shape[1]!==dims.height)||(typeof input.shape[2]==='number'&&input.shape[2]!==dims.width))throw new Error('ONNX observation tensor shape does not match checkpoint dimensions/schema');
  if(!logits?.isTensor||logits.type!=='float32'||logits.shape?.at(-1)!==4)throw new Error('ONNX logits output must be a float32 tensor with four actions');
}
function toWire(state:ArenaState) {
  return {width:state.width,height:state.height,positions:state.agents.map(agent=>agent.position),headings:state.agents.map(agent=>['UP','RIGHT','DOWN','LEFT'][agent.heading]),trails:state.agents.map(agent=>agent.trail)};
}
export async function validateReplayArtifact(replay:ReplayArtifact):Promise<ArenaState[]> {
  if(replay.format!=='lc-replay/1'||replay.engine_version!=='lc-engine/1'||replay.observation_schema!=='lc-observation/1'||replay.action_schema!=='lc-action/1')throw new Error('Unsupported replay, engine, or observation/action schema');
  if(!(await verifyObjectDigest(replay as any,'replay_sha256')))throw new Error('Replay digest is invalid');
  if(typeof replay.master_seed!=='string'||!/^(0|[1-9][0-9]*)$/.test(replay.master_seed)||!Number.isSafeInteger(replay.arena_index)||replay.arena_index<0)throw new Error('Replay master seed or arena index is invalid');
  if(!Array.isArray(replay.ticks)||!Array.isArray(replay.agents)||replay.tick_count!==replay.ticks.length||replay.agents.length!==2||!Array.isArray(replay.derived_streams)||replay.derived_streams.length!==3)throw new Error('Replay length, stream, or agent count is invalid');
  const initial=replay.initial_state;
  if(!initial||!Number.isSafeInteger(initial.width)||!Number.isSafeInteger(initial.height)||initial.width<2||initial.height<2)throw new Error('Replay initial board dimensions are invalid');
  const deterministic=await createSeededState({width:initial.width,height:initial.height,masterSeed:replay.master_seed,arenaIndex:replay.arena_index});
  const expectedStreams=[{label:'episode',initstate:String(deterministic.streams.episode.initstate),initseq:String(deterministic.streams.episode.initseq)},{label:'agent/0',initstate:String(deterministic.streams.agents[0].initstate),initseq:String(deterministic.streams.agents[0].initseq)},{label:'agent/1',initstate:String(deterministic.streams.agents[1].initstate),initseq:String(deterministic.streams.agents[1].initseq)}];
  if(canonicalJson(replay.derived_streams)!==canonicalJson(expectedStreams)||canonicalJson(replay.initial_state)!==canonicalJson(toWire(deterministic.state)))throw new Error('Replay initial state or derived streams do not match its deterministic seed');
  const rebuilt=replayActions(deterministic.state,replay.ticks.map(t=>t.joint_actions as [JointAction,JointAction]));
  if(rebuilt.result===null||rebuilt.result!==replay.result||rebuilt.records.length!==replay.tick_count)throw new Error('Replay does not reach its declared terminal result');
  if(replay.ticks.some((t,i)=>t.tick!==i||canonicalJson(t.collision_causes)!==canonicalJson(rebuilt.records[i].collision_causes)||canonicalJson(t.terminal_causes)!==canonicalJson(rebuilt.records[i].terminal_causes)))throw new Error('Replay tick, collision, or terminal causes differ from the engine');
  return rebuilt.states;
}
export async function validateEvaluationReport(report:any):Promise<any> {
  if(report?.format!=='lc-eval/1'||!Array.isArray(report.matchups))throw new Error('Expected an lc-eval/1 evaluation report');
  if(!(await verifyObjectDigest(report,'report_sha256')))throw new Error('Evaluation report digest is invalid');
  const fail=()=>{throw new Error('Evaluation report does not satisfy the complete lc-eval/1 benchmark contract')};
  const safePath=(path:any)=>typeof path==='string'&&path.length>0&&!path.startsWith('/')&&!path.includes('\\')&&!path.includes('\0')&&!path.split('/').some((part:string)=>part==='..'||part==='.'||part==='');
  const close=(a:number,b:number)=>Math.abs(a-b)<=1e-12;
  if(!hasOnlyKeys(report,['format','run_id','checkpoint_id','protocol','held_out_seeds','matchups','report_sha256'])||!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(report.run_id??'')||!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(report.checkpoint_id??'')||report.protocol!=='lc-eval/1')fail();
  const seeds=Array.from({length:32},(_,i)=>String(i));
  if(canonicalJson(report.held_out_seeds)!==canonicalJson(seeds)||report.matchups.length!==2)fail();
  const byOpponent=new Map(report.matchups.map((matchup:any)=>[matchup?.opponent_id,matchup]));
  if(byOpponent.size!==2||!byOpponent.has('open-space')||!byOpponent.has('step-0'))fail();
  const replayPaths=new Set<string>();
  const wilson=(wins:number,draws:number,losses:number)=>{const n=wins+draws+losses,p=wins/n,z=1.959963984540054,denominator=1+z*z/n,center=(p+z*z/(2*n))/denominator,margin=z*Math.sqrt((p*(1-p)/n+z*z/(4*n*n)))/denominator;return [Math.max(0,center-margin),Math.min(1,center+margin)]};
  for(const opponent of ['open-space','step-0']){
    const matchup:any=byOpponent.get(opponent),games=matchup?.games,summary=matchup?.summary;
    if(!hasOnlyKeys(matchup,['opponent_id','games','summary'])||!Array.isArray(games)||games.length!==64||!hasOnlyKeys(summary,['wins','draws','losses','win_rate','wilson_95','mean_survival_ticks','median_survival_ticks','collision_causes']))fail();
    const seen=new Set<string>(),ticks:number[]=[],causes:Record<string,number>={},counts={win:0,draw:0,loss:0};
    for(const game of games){
      if(!hasOnlyKeys(game,['seed','target_seat','result','ticks','collision_causes','replay_file'])||typeof game.seed!=='string'||!seeds.includes(game.seed)||![0,1].includes(game.target_seat)||!['win','draw','loss'].includes(game.result)||!Number.isSafeInteger(game.ticks)||game.ticks<0||!Array.isArray(game.collision_causes)||new Set(game.collision_causes).size!==game.collision_causes.length||game.collision_causes.some((cause:string)=>!['out_of_bounds','occupied_trail','occupied_head','same_destination','head_swap','invalid_action','reverse_action'].includes(cause))||!safePath(game.replay_file))fail();
      const key=`${game.seed}/${game.target_seat}`;if(seen.has(key)||replayPaths.has(game.replay_file))fail();seen.add(key);replayPaths.add(game.replay_file);counts[game.result as keyof typeof counts]++;ticks.push(game.ticks);for(const cause of game.collision_causes)causes[cause]=(causes[cause]??0)+1;
    }
    if(seen.size!==64)fail();
    const sorted=[...ticks].sort((a,b)=>a-b),mean=ticks.reduce((sum,value)=>sum+value,0)/64,median=(sorted[31]+sorted[32])/2,interval=wilson(counts.win,counts.draw,counts.loss);
    if(!Number.isSafeInteger(summary.wins)||!Number.isSafeInteger(summary.draws)||!Number.isSafeInteger(summary.losses)||summary.wins!==counts.win||summary.draws!==counts.draw||summary.losses!==counts.loss||summary.wins+summary.draws+summary.losses!==64||!Number.isFinite(summary.win_rate)||!close(summary.win_rate,counts.win/64)||!Number.isFinite(summary.mean_survival_ticks)||!close(summary.mean_survival_ticks,mean)||!Number.isFinite(summary.median_survival_ticks)||!close(summary.median_survival_ticks,median)||!Array.isArray(summary.wilson_95)||summary.wilson_95.length!==2||!summary.wilson_95.every(Number.isFinite)||!close(summary.wilson_95[0],interval[0])||!close(summary.wilson_95[1],interval[1])||canonicalJson(summary.collision_causes)!==canonicalJson(causes))fail();
  }
  const matchup=byOpponent.get('open-space') as any,summary=matchup.summary;
  return {matchup,summary};
}

export async function validateRunManifest(manifest:any):Promise<void> {
  if(manifest?.format!=='lc-run/1'||typeof manifest.run_id!=='string'||!Array.isArray(manifest.checkpoints)||!manifest.evaluation||manifest.evaluation.protocol!=='lc-eval/1')throw new Error('Expected a complete lc-run/1 manifest');
  if(!/^[a-f0-9]{64}$/.test(manifest.manifest_sha256??'')||!(await verifyObjectDigest(manifest,'manifest_sha256')))throw new Error('Run manifest digest is invalid');
  if(manifest.observation_schema!=='lc-observation/1'||manifest.action_schema!=='lc-action/1'||manifest.replay_schema!=='lc-replay/1')throw new Error('Run manifest declares unsupported artifact schemas');
  const paths=[...(manifest.files??[]).map((entry:any)=>entry.path),...(manifest.checkpoints??[]).map((entry:any)=>entry.file),manifest.evaluation.metrics_file,...(manifest.evaluation.replay_files??[])];for(const path of paths){if(typeof path!=='string'||path.startsWith('/')||path.split('/').includes('..')||path.includes('\\'))throw new Error('Run manifest contains an unsafe artifact path')}
}

export type TrainingMetrics={trainingSteps:number;updates:number;latestLoss:number;environmentStepsPerSecond:number;selectedDevice:string;cudaOptimizerUpdates:number};
export type VerifiedRunBundle={manifest:any;report:any;training:TrainingMetrics;replays:Map<string,ReplayArtifact>};
async function sha256Hex(bytes:Uint8Array):Promise<string>{return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes.slice().buffer as ArrayBuffer)),byte=>byte.toString(16).padStart(2,'0')).join('')}
export type BundleFile=Uint8Array|Blob;
async function bundleBytes(file:BundleFile):Promise<Uint8Array>{return file instanceof Uint8Array?file:new Uint8Array(await file.arrayBuffer())}
async function parseJsonFile(files:ReadonlyMap<string,BundleFile>,path:string):Promise<any> {
  const file=files.get(path);if(!file)throw new Error(`Run bundle is missing required file: ${path}`);
  try{return JSON.parse(new TextDecoder().decode(await bundleBytes(file)))}catch{throw new Error(`Run bundle file is not valid JSON: ${path}`)}
}
function validateTrainingMetrics(value:any,runId:string):TrainingMetrics {
  const training=value?.training,device=value?.device,updates=training?.updates;
  if(value?.run_id!==runId||!Number.isSafeInteger(training?.training_steps)||training.training_steps<0||!Array.isArray(updates)||updates.length===0||typeof device?.device!=='string'||!device.device||!Number.isSafeInteger(device.optimizer_updates)||device.optimizer_updates<0||!Number.isSafeInteger(device.cuda_optimizer_updates)||device.cuda_optimizer_updates<0)throw new Error('Training metrics have an invalid run/device/update summary');
  if(!Number.isFinite(training.environment_steps_per_second)||training.environment_steps_per_second<0||!Number.isFinite(training.mean_update_latency_seconds)||training.mean_update_latency_seconds<0)throw new Error('Training metrics have invalid throughput or update latency');
  for(const update of updates)if(!Number.isSafeInteger(update.update)||update.update<1||!Number.isSafeInteger(update.optimizer_updates)||update.optimizer_updates<0||!Number.isFinite(update.mean_loss)||!Number.isFinite(update.update_latency_seconds)||update.update_latency_seconds<0||!Number.isSafeInteger(update.cuda_updates)||update.cuda_updates<0)throw new Error('Training metrics contain an invalid update record');
  const latest=updates[updates.length-1];
  if(latest.optimizer_updates>device.optimizer_updates||latest.cuda_updates!==device.cuda_optimizer_updates)throw new Error('Training metrics update/device totals are inconsistent');
  return {trainingSteps:training.training_steps,updates:updates.length,latestLoss:latest.mean_loss,environmentStepsPerSecond:training.environment_steps_per_second,selectedDevice:device.device,cudaOptimizerUpdates:device.cuda_optimizer_updates};
}
export async function validateRunBundle(manifest:any,files:ReadonlyMap<string,BundleFile>):Promise<VerifiedRunBundle> {
  await validateRunManifest(manifest);
  if(!Array.isArray(manifest.files)||!Array.isArray(manifest.checkpoints)||!Array.isArray(manifest.evaluation?.replay_files))throw new Error('Run bundle manifest is missing required file/checkpoint/evaluation lists');
  const listed=new Map<string,string>();
  for(const entry of manifest.files){if(!entry||typeof entry.path!=='string'||listed.has(entry.path))throw new Error('Run manifest contains an invalid or duplicate file entry');listed.set(entry.path,entry.sha256);const file=files.get(entry.path);if(!file)throw new Error(`Run bundle is missing required file: ${entry.path}`);if(await sha256Hex(await bundleBytes(file))!==entry.sha256)throw new Error(`Run bundle SHA-256 mismatch: ${entry.path}`)}
  const requiredFile=(path:string)=>{if(!listed.has(path)||!files.has(path))throw new Error(`Run manifest does not protect required file: ${path}`)};
  const groups=new Map<string,any[]>();
  for(const checkpoint of manifest.checkpoints){requiredFile(checkpoint.file);if(listed.get(checkpoint.file)!==checkpoint.sha256)throw new Error(`Checkpoint reference hash does not match run file hash: ${checkpoint.file}`);if(checkpoint.format==='onnx'&&files.has(`${checkpoint.file}.data`))throw new Error(`Viewer requires single-file ONNX; external-data sidecar is unsupported: ${checkpoint.file}.data`);const checkpointFile=files.get(checkpoint.file)!;if((checkpointFile instanceof Uint8Array?checkpointFile.byteLength:checkpointFile.size)===0)throw new Error(`Run bundle checkpoint is empty: ${checkpoint.file}`);const key=`${checkpoint.checkpoint_id}\0${checkpoint.step}`;groups.set(key,[...(groups.get(key)??[]),checkpoint])}
  const checkpointIds=new Set<string>();
  for(const group of groups.values()){
    if(group.length!==2||new Set(group.map((entry:any)=>entry.format)).size!==2||!group.some((entry:any)=>entry.format==='onnx')||!group.some((entry:any)=>entry.format==='pytorch'))throw new Error('Each checkpoint must have one ONNX and one PyTorch file');
    const onnx=group.find((entry:any)=>entry.format==='onnx'),pytorch=group.find((entry:any)=>entry.format==='pytorch');checkpointIds.add(onnx.checkpoint_id);
    const parent=onnx.file.includes('/')?onnx.file.slice(0,onnx.file.lastIndexOf('/')+1):'',metadataPath=`${parent}${onnx.checkpoint_id}.metadata.json`;requiredFile(metadataPath);
    const metadata=await parseJsonFile(files,metadataPath) as CheckpointMetadata;await validateCheckpointMetadata(metadata);
    if(metadata.checkpoint_id!==onnx.checkpoint_id||metadata.step!==onnx.step||metadata.onnx.file!==onnx.file||metadata.onnx.sha256!==onnx.sha256||metadata.pytorch.file!==pytorch.file||metadata.pytorch.sha256!==pytorch.sha256||onnx.observation_schema!==metadata.observation_schema||onnx.action_schema!==metadata.action_schema||pytorch.observation_schema!==metadata.observation_schema||pytorch.action_schema!==metadata.action_schema)throw new Error(`Checkpoint metadata does not match manifest refs: ${metadataPath}`);
  }
  const metricsPath=manifest.evaluation.metrics_file;requiredFile(metricsPath);const report=await parseJsonFile(files,metricsPath);await validateEvaluationReport(report);
  if(report.run_id!==manifest.run_id||!checkpointIds.has(report.checkpoint_id))throw new Error('Evaluation report identity does not match the run bundle');
  const trainingPath=manifest.files.find((entry:any)=>entry.path==='training-metrics.json')?.path;if(!trainingPath)throw new Error('Run bundle manifest does not protect training-metrics.json');
  const training=validateTrainingMetrics(await parseJsonFile(files,trainingPath),manifest.run_id);
  const expectedReplayPaths=new Set<string>();for(const matchup of report.matchups)for(const game of matchup.games)expectedReplayPaths.add(game.replay_file);
  if(manifest.evaluation.replay_files.length!==expectedReplayPaths.size||new Set(manifest.evaluation.replay_files).size!==manifest.evaluation.replay_files.length||manifest.evaluation.replay_files.some((path:string)=>!expectedReplayPaths.has(path))||Array.from(expectedReplayPaths).some(path=>!manifest.evaluation.replay_files.includes(path)))throw new Error('Run manifest replay list does not exactly match evaluation replay references');
  const replays=new Map<string,ReplayArtifact>();
  for(const path of manifest.evaluation.replay_files){requiredFile(path);const replay=await parseJsonFile(files,path) as ReplayArtifact;await validateReplayArtifact(replay);if(replay.run_id!==manifest.run_id)throw new Error(`Replay run_id does not match the bundle: ${path}`);replays.set(path,replay)}
  for(const matchup of report.matchups)for(const game of matchup.games){const replay=replays.get(game.replay_file)!;const targetSeat=game.target_seat,expectedResult=replay.result==='draw'?'draw':replay.result===`agent_${targetSeat}_win`?'win':'loss',replayCauses=[...new Set(replay.ticks.flatMap(tick=>tick.collision_causes.flat()))].sort();if(replay.master_seed!==game.seed||replay.arena_index!==Number(game.seed)||replay.tick_count!==game.ticks||expectedResult!==game.result||canonicalJson(game.collision_causes)!==canonicalJson(replayCauses)||replay.match_id!==`${matchup.opponent_id}-${game.seed.padStart(2,'0')}-${targetSeat}`||replay.agents[targetSeat]?.checkpoint_id!==report.checkpoint_id||replay.agents[targetSeat===0?1:0]?.checkpoint_id!==matchup.opponent_id)throw new Error(`Evaluation game does not match its validated replay: ${game.replay_file}`)}
  return {manifest,report,training,replays};
}

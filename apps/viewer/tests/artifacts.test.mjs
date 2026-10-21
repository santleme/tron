import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { validateCheckpointMetadata, assertCheckpointMetadataPresent, assertModelMatchesMetadata, validateReplayArtifact, validateEvaluationReport, validateRunBundle, canonicalJson } from '../src/artifacts.ts';
function digest(value,field){const copy={...value};delete copy[field];return createHash('sha256').update(canonicalJson(copy)).digest('hex')}
function checkpointMetadata(){
 const metadata={format:'lc-checkpoint/1',checkpoint_id:'step-24',step:24,model_architecture:{name:'mlp-hwc-flatten',version:'1',config:{width:10,height:14,hidden_size:64}},observation_schema:'lc-observation/1',action_schema:'lc-action/1',onnx:{file:'checkpoints/step-24.onnx',sha256:'a'.repeat(64),opset:17,runtime:'onnxruntime-web'},pytorch:{file:'checkpoints/step-24.pt',sha256:'b'.repeat(64),format:'state_dict+optimizer'},artifacts:[{format:'onnx',file:'checkpoints/step-24.onnx',sha256:'a'.repeat(64)},{format:'pytorch',file:'checkpoints/step-24.pt',sha256:'b'.repeat(64)}]};metadata.metadata_sha256=digest(metadata,'metadata_sha256');return metadata;
}
test('checkpoint metadata validates full contract and supports declared model dimensions',async()=>{
 const m=checkpointMetadata();await validateCheckpointMetadata(m);
 const v2=structuredClone(m);v2.model_architecture.version='2';v2.metadata_sha256=digest(v2,'metadata_sha256');await validateCheckpointMetadata(v2);
 const unsupported=structuredClone(v2);unsupported.model_architecture.version='3';unsupported.metadata_sha256=digest(unsupported,'metadata_sha256');await assert.rejects(()=>validateCheckpointMetadata(unsupported),/Unsupported checkpoint model architecture/);
 await assert.rejects(()=>validateCheckpointMetadata({...m,onnx:{...m.onnx,opset:18}}),/Unsupported ONNX opset/);
 await assert.rejects(()=>validateCheckpointMetadata({...m,metadata_sha256:'0'.repeat(64)}),/digest/);
 const unknown={...m,unmodeled:true};unknown.metadata_sha256=digest(unknown,'metadata_sha256');await assert.rejects(()=>validateCheckpointMetadata(unknown),/unknown fields/);
 const noConfig={...m,model_architecture:{name:m.model_architecture.name,version:m.model_architecture.version}};noConfig.metadata_sha256=digest(noConfig,'metadata_sha256');await assert.rejects(()=>validateCheckpointMetadata(noConfig),/architecture config/);
 const unsafe={...m,onnx:{...m.onnx,file:'../model.onnx'}};unsafe.metadata_sha256=digest(unsafe,'metadata_sha256');await assert.rejects(()=>validateCheckpointMetadata(unsafe),/file path/);
 const mismatched={...m,artifacts:m.artifacts.map(a=>a.format==='onnx'?{...a,sha256:'c'.repeat(64)}:a)};mismatched.metadata_sha256=digest(mismatched,'metadata_sha256');await assert.rejects(()=>validateCheckpointMetadata(mismatched),/does not match/);
});
test('ONNX metadata must match loaded checkpoint bytes, interface, and tensor dimensions',()=>{
 const m=checkpointMetadata(), inputs=['observation'],outputs=['logits','value'];
 const inputMeta=[{name:'observation',isTensor:true,type:'float32',shape:['batch',14,10,12]}],outputMeta=[{name:'logits',isTensor:true,type:'float32',shape:['batch',4]}];
 assertModelMatchesMetadata(m,'step-24.onnx','a'.repeat(64),inputs,outputs,inputMeta,outputMeta);
 assert.throws(()=>assertModelMatchesMetadata(m,'step-24.onnx','c'.repeat(64),inputs,outputs,inputMeta,outputMeta),/SHA-256/);
 assert.throws(()=>assertModelMatchesMetadata(m,'renamed.onnx','a'.repeat(64),inputs,outputs,inputMeta,outputMeta),/filename/);
 assert.throws(()=>assertModelMatchesMetadata(m,'step-24.onnx','a'.repeat(64),inputs,outputs,[{...inputMeta[0],shape:['batch',12,12,12]}],outputMeta),/dimensions/);
});
function wilson(wins,draws,losses){const n=wins+draws+losses,p=wins/n,z=1.959963984540054,d=1+z*z/n,c=(p+z*z/(2*n))/d,m=z*Math.sqrt((p*(1-p)/n+z*z/(4*n*n)))/d;return [Math.max(0,c-m),Math.min(1,c+m)]}
function fullReportFromGames(matchups,runId='run-1',checkpointId='step-1'){const report={format:'lc-eval/1',run_id:runId,checkpoint_id:checkpointId,protocol:'lc-eval/1',held_out_seeds:Array.from({length:32},(_,i)=>String(i)),matchups:matchups.map(({opponent_id,games})=>{const wins=games.filter(g=>g.result==='win').length,draws=games.filter(g=>g.result==='draw').length,losses=64-wins-draws,ticks=games.map(g=>g.ticks).sort((a,b)=>a-b),causes={};for(const game of games)for(const cause of game.collision_causes)causes[cause]=(causes[cause]??0)+1;return {opponent_id,games,summary:{wins,draws,losses,win_rate:wins/64,wilson_95:wilson(wins,draws,losses),mean_survival_ticks:games.reduce((sum,g)=>sum+g.ticks,0)/64,median_survival_ticks:(ticks[31]+ticks[32])/2,collision_causes:causes}}})};report.report_sha256=digest(report,'report_sha256');return report}
function syntheticReport(){
 const matchups=['open-space','step-0'].map(opponent_id=>({opponent_id,games:Array.from({length:64},(_,i)=>{const seed=String(Math.floor(i/2)),target_seat=i%2;return {seed,target_seat,result:i%4===0?'win':i%4===1?'draw':'loss',ticks:i+1,collision_causes:i%2?['out_of_bounds']:[],replay_file:`evaluation/${opponent_id}-${seed.padStart(2,'0')}-${target_seat}.json`}})}));
 return fullReportFromGames(matchups);
}
test('evaluation report reader enforces full 64-game benchmark contract and summary consistency',async()=>{
 const report=syntheticReport();assert.equal((await validateEvaluationReport(report)).summary.wins,16);
 await assert.rejects(()=>validateEvaluationReport({...report,format:'other'}),/lc-eval\/1/);
 await assert.rejects(()=>validateEvaluationReport({format:'lc-eval/1',matchups:[{}]}),/digest/);
 const incomplete=structuredClone(report);incomplete.matchups.pop();incomplete.report_sha256=digest(incomplete,'report_sha256');await assert.rejects(()=>validateEvaluationReport(incomplete),/complete lc-eval/);
 const missingSeat=structuredClone(report);missingSeat.matchups[0].games.pop();missingSeat.report_sha256=digest(missingSeat,'report_sha256');await assert.rejects(()=>validateEvaluationReport(missingSeat),/complete lc-eval/);
 const mismatchedSummary=structuredClone(report);mismatchedSummary.matchups[0].summary.wins++;mismatchedSummary.report_sha256=digest(mismatchedSummary,'report_sha256');await assert.rejects(()=>validateEvaluationReport(mismatchedSummary),/complete lc-eval/);
 const corrupt={...report,report_sha256:'0'.repeat(64)};await assert.rejects(()=>validateEvaluationReport(corrupt),/digest/);
});
import { readFile } from 'node:fs/promises';
import { validateRunManifest } from '../src/artifacts.ts';
import { createSeededState, step } from '../engine/engine.ts';
async function deterministicReplay({seed='123456789',arenaIndex=4,runId='test-run',matchId='deterministic-seed',targetSeat=0,opponent='step-0'}={}){
 const init=await createSeededState({width:8,height:6,masterSeed:seed,arenaIndex});let state=init.state;const ticks=[];while(state.result===null){const transition=step(state,[state.agents[0].heading,state.agents[1].heading]);ticks.push(transition.record);state=transition.state}
 const agents=Array.from({length:2},(_,seat)=>({agent_id:seat===targetSeat?'target':opponent,checkpoint_id:seat===targetSeat?'step-1':opponent}));
 const replay={format:'lc-replay/1',run_id:runId,match_id:matchId,engine_version:'lc-engine/1',observation_schema:'lc-observation/1',action_schema:'lc-action/1',master_seed:init.master_seed,arena_index:init.arena_index,derived_streams:[{label:'episode',initstate:String(init.streams.episode.initstate),initseq:String(init.streams.episode.initseq)},{label:'agent/0',initstate:String(init.streams.agents[0].initstate),initseq:String(init.streams.agents[0].initseq)},{label:'agent/1',initstate:String(init.streams.agents[1].initstate),initseq:String(init.streams.agents[1].initseq)}],agents,initial_state:{width:8,height:6,positions:init.state.agents.map(a=>a.position),headings:init.state.agents.map(a=>['UP','RIGHT','DOWN','LEFT'][a.heading]),trails:init.state.agents.map(a=>a.trail)},ticks,result:state.result,tick_count:ticks.length,replay_sha256:''};replay.replay_sha256=digest(replay,'replay_sha256');return replay;
}
test('replay validates deterministic seed, derived streams, initial trails, collision and terminal causes',async()=>{
 const replay=await deterministicReplay(),states=await validateReplayArtifact(replay);assert.equal(states.length,replay.tick_count+1);assert.equal(states.at(-1).result,replay.result);
 for(const mutate of [r=>{r.master_seed='987654321'},r=>{r.arena_index+=1},r=>{r.derived_streams[0].initstate='1'},r=>{r.initial_state.trails[0][0].x=(r.initial_state.trails[0][0].x+1)%r.initial_state.width},r=>{r.ticks.at(-1).terminal_causes=[]},r=>{r.ticks[0].tick=4}]){const malformed=structuredClone(replay);mutate(malformed);malformed.replay_sha256=digest(malformed,'replay_sha256');await assert.rejects(()=>validateReplayArtifact(malformed),/derived streams|terminal causes|tick, collision/)}
});
test('run manifest validates digest, schema versions, and artifact paths',async()=>{
 const manifest=JSON.parse(await readFile(new URL('../../../shared/fixtures/lc-run-1.valid.json',import.meta.url),'utf8'));
 await validateRunManifest(manifest);
 const unsafe=structuredClone(manifest);unsafe.checkpoints[0].file='../secret';unsafe.manifest_sha256=digest(unsafe,'manifest_sha256');await assert.rejects(()=>validateRunManifest(unsafe),/unsafe artifact path/);
});

test('viewer constrains outbound requests to local assets',async()=>{
 const html=await readFile(new URL('../index.html',import.meta.url),'utf8'),source=await readFile(new URL('../src/main.ts',import.meta.url),'utf8');
 assert.match(html,/connect-src 'self' blob:/);assert.doesNotMatch(html,/https?:\/\//);assert.doesNotMatch(source,/\bfetch\s*\(/);
});

test('Python lc-eval decimal fixture digest is accepted cross-runtime',async()=>{
 const report=JSON.parse(await readFile(new URL('../../../shared/fixtures/lc-eval-float-1.json',import.meta.url),'utf8'));
 assert.equal(report.matchups[0].summary.mean_survival_ticks,2.546875);await validateEvaluationReport(report);assert.equal(report.matchups.find(matchup=>matchup.opponent_id==='step-0').summary.win_rate,0.40625);
});

test('metadata is mandatory before a playable model can be created',()=>{
 assert.throws(()=>assertCheckpointMetadataPresent(null),/metadata before its ONNX model/);assert.doesNotThrow(()=>assertCheckpointMetadataPresent(checkpointMetadata()));
});

function bytes(value){return value instanceof Uint8Array?value:new TextEncoder().encode(typeof value==='string'?value:JSON.stringify(value))}
function fileSha(content){return createHash('sha256').update(content).digest('hex')}
async function completeTestBundle(reportRun='test-run'){
 const files=new Map(),entries=[],replayFiles=[],matchups=[];
 const add=(path,value)=>{const data=bytes(value);files.set(path,data);entries.push({path,sha256:fileSha(data)})};
 const metadata=checkpointMetadata();metadata.checkpoint_id='step-1';metadata.step=1;metadata.onnx.file='checkpoints/step-1.onnx';metadata.pytorch.file='checkpoints/step-1.pt';metadata.artifacts=[{format:'onnx',file:metadata.onnx.file,sha256:metadata.onnx.sha256},{format:'pytorch',file:metadata.pytorch.file,sha256:metadata.pytorch.sha256}];metadata.metadata_sha256=digest(metadata,'metadata_sha256');
 const onnx=bytes('onnx-placeholder'),pytorch=bytes('pt-placeholder');metadata.onnx.sha256=fileSha(onnx);metadata.pytorch.sha256=fileSha(pytorch);metadata.artifacts[0].sha256=metadata.onnx.sha256;metadata.artifacts[1].sha256=metadata.pytorch.sha256;metadata.metadata_sha256=digest(metadata,'metadata_sha256');
 const matchupGames=[];for(const opponent of ['open-space','step-0']){const games=[];for(let seed=0;seed<32;seed++)for(const targetSeat of [0,1]){const matchId=`${opponent}-${String(seed).padStart(2,'0')}-${targetSeat}`,path=`evaluation/replays/${matchId}.json`,replay=await deterministicReplay({seed:String(seed),arenaIndex:seed,runId:'test-run',matchId,targetSeat,opponent});const result=replay.result==='draw'?'draw':replay.result===`agent_${targetSeat}_win`?'win':'loss',collisionCauses=[...new Set(replay.ticks.flatMap(t=>t.collision_causes.flat()))].sort();games.push({seed:String(seed),target_seat:targetSeat,result,ticks:replay.tick_count,collision_causes:collisionCauses,replay_file:path});replayFiles.push(path);add(path,replay)}matchupGames.push({opponent_id:opponent,games})}
 const report=fullReportFromGames(matchupGames,reportRun,'step-1');
 const training={run_id:'test-run',training:{training_steps:4096,environment_steps_per_second:123.5,mean_update_latency_seconds:0.25,updates:[{update:1,optimizer_updates:4,mean_loss:0.75,update_latency_seconds:0.25,cuda_updates:4}]},device:{device:'cuda:0',optimizer_updates:4,cuda_optimizer_updates:4}};
 add(metadata.onnx.file,onnx);add(metadata.pytorch.file,pytorch);add('checkpoints/step-1.metadata.json',metadata);add('evaluation/report.json',report);add('training-metrics.json',training);
 const manifest={format:'lc-run/1',run_id:'test-run',created_utc:'2026-10-19T00:00:00Z',engine_version:'lc-engine/1',observation_schema:'lc-observation/1',action_schema:'lc-action/1',reward_schema:'lc-reward/1',replay_schema:'lc-replay/1',master_seed:'123456789',training_config:{},training_steps:4096,checkpoints:[{checkpoint_id:'step-1',step:1,file:metadata.pytorch.file,sha256:metadata.pytorch.sha256,format:'pytorch',observation_schema:'lc-observation/1',action_schema:'lc-action/1'},{checkpoint_id:'step-1',step:1,file:metadata.onnx.file,sha256:metadata.onnx.sha256,format:'onnx',observation_schema:'lc-observation/1',action_schema:'lc-action/1'}],evaluation:{protocol:'lc-eval/1',held_out_seeds:Array.from({length:32},(_,i)=>String(i)),games_per_matchup:64,opponent_ids:['open-space','step-0'],metrics_file:'evaluation/report.json',replay_files:replayFiles},software:{},device:{device:'cuda:0'},benchmark_records:[],files:entries};manifest.manifest_sha256=digest(manifest,'manifest_sha256');return {manifest,files};
}
test('local bundle verifies all linked checkpoint, eval, replay and training artifacts',async()=>{
 const {manifest,files}=await completeTestBundle();const browserFiles=new Map(Array.from(files,([path,content])=>[path,new Blob([content])]));const bundle=await validateRunBundle(manifest,browserFiles);assert.equal(bundle.replays.size,128);assert.equal(bundle.training.trainingSteps,4096);assert.equal(bundle.training.environmentStepsPerSecond,123.5);assert.equal(bundle.training.selectedDevice,'cuda:0');assert.equal(bundle.training.cudaOptimizerUpdates,4);assert.equal(bundle.report.run_id,'test-run');
});
test('local bundle fails closed for missing/hash-mismatched files and wrong eval identity',async()=>{
 const missing=await completeTestBundle();missing.files.delete('training-metrics.json');await assert.rejects(()=>validateRunBundle(missing.manifest,missing.files),/missing required file/);
 const sidecar=await completeTestBundle();sidecar.files.set('checkpoints/step-1.onnx.data',bytes('external weights'));await assert.rejects(()=>validateRunBundle(sidecar.manifest,sidecar.files),/external-data sidecar is unsupported/);
 const listedSidecar=await completeTestBundle();const sidecarBytes=bytes('external weights');listedSidecar.files.set('checkpoints/step-1.onnx.data',sidecarBytes);listedSidecar.manifest.files.push({path:'checkpoints/step-1.onnx.data',sha256:fileSha(sidecarBytes)});listedSidecar.manifest.manifest_sha256=digest(listedSidecar.manifest,'manifest_sha256');await assert.rejects(()=>validateRunBundle(listedSidecar.manifest,listedSidecar.files),/external-data sidecar is unsupported/);
 const corrupted=await completeTestBundle();corrupted.files.set('checkpoints/step-1.onnx',bytes('wrong onnx'));await assert.rejects(()=>validateRunBundle(corrupted.manifest,corrupted.files),/SHA-256 mismatch/);
 const wrongIdentity=await completeTestBundle('some-other-run');await assert.rejects(()=>validateRunBundle(wrongIdentity.manifest,wrongIdentity.files),/Evaluation report identity/);
 const replayLinkMismatch=await completeTestBundle();replayLinkMismatch.manifest.evaluation.replay_files.pop();replayLinkMismatch.manifest.manifest_sha256=digest(replayLinkMismatch.manifest,'manifest_sha256');await assert.rejects(()=>validateRunBundle(replayLinkMismatch.manifest,replayLinkMismatch.files),/replay list does not exactly match/);
});

test('Python-generated replay contract passes TypeScript seed/replay validation',async()=>{
 const replay=JSON.parse(await readFile(new URL('./fixtures/python-replay-valid.json',import.meta.url),'utf8'));const states=await validateReplayArtifact(replay);assert.equal(states.length,replay.tick_count+1);assert.equal(states.at(-1).result,replay.result);
});

import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
// Supply the paths returned by Codex load_workspace_dependencies and the presentation skill.
// RUNTIME_NODE_MODULES, RUNTIME_PYTHON and PRESENTATION_SKILL_DIR are required.
// PRESENTATION_OUTPUT may name a new output path because finalization never overwrites a deck.
const modules=process.env.RUNTIME_NODE_MODULES;
if(!modules||!process.env.RUNTIME_PYTHON||!process.env.PRESENTATION_SKILL_DIR)throw new Error('Set RUNTIME_NODE_MODULES, RUNTIME_PYTHON and PRESENTATION_SKILL_DIR from the Codex bundled runtime.');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(modules,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const WORK=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const OUT=path.join(WORK,'submission');
const SKILL=process.env.PRESENTATION_SKILL_DIR;
const PY=process.env.RUNTIME_PYTHON;
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const metrics=JSON.parse(await fs.readFile(path.join(OUT,'results/metrics.json'),'utf8'));
const parity=JSON.parse(await fs.readFile(path.join(OUT,'results/browser_parity.json'),'utf8'));
const get=(scenario,model,policy='validation_f1',split='test')=>metrics.find(r=>r.scenario===scenario&&r.model===model&&r.policy===policy&&r.split===split);
const names=['LogisticRegression','RandomForest','XGBoost','SVM','GradientBoosting','MLP','ResidualMLP','CNN1D'];
const labels=['Logistic Regression','Random Forest','XGBoost','SVM','Histogram GB','MLP','Residual MLP','1D CNN'];
const short=['LR','RF','XGB','SVM','HGB','MLP','ResMLP','CNN'];
const F=v=>Number(v).toFixed(4),P=v=>(v*100).toFixed(1)+'%';
const pres=Presentation.create({slideSize:{width:1280,height:720}});
pres.theme.defaultFont='Arial';
const C={bg:'#FAF9F5',ink:'#182B32',muted:'#5F7177',accent:'#D95C31',blue:'#315B68',line:'#D7DFDA',pale:'#EDF2EF'};
function text(s,t,l,top,w,height,size=25,color=C.ink,bold=false){const sh=s.shapes.add({geometry:'textbox',position:{left:l,top,width:w,height},fill:'none',line:{fill:'none',width:0}});sh.text=t;sh.text.style={typeface:'Arial',fontSize:size,color,bold};return sh;}
function slide(title,subtitle='',notes=''){
 const s=pres.slides.add();s.background.fill=C.bg;
 text(s,title,64,47,1150,78,43,C.ink,true);
 if(subtitle)text(s,subtitle,67,127,1130,48,23,C.muted);
 text(s,String(pres.slides.items.length).padStart(2,'0'),1170,672,48,25,16,C.muted);
 s.speakerNotes.text=notes+'\nEvidence: submission/results/metrics.csv and predictions.csv. Measured 20 October 2026.';return s;
}
function para(s,t,top=205,w=1115,size=28,left=70,height=100){return text(s,t,left,top,w,height,size);}
function table(s,headers,values,{top=200,height=365,width=1144,left=68,widths}={}){
 const t=s.tables.add({rows:values.length+1,columns:headers.length,left,top,width,height,values:[headers,...values],columnWidths:widths});
 t.borders.assign({style:'solid',fill:C.line,width:1});t.styleOptions={headerRow:true,bandedRows:true};
 for(let r=0;r<values.length+1;r++)for(let c=0;c<headers.length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?C.blue:(r%2?C.bg:C.pale);
  cell.text.style={typeface:'Arial',fontSize:23,color:r===0?'#FFFFFF':C.ink,bold:r===0};
 }
 return t;
}
function chart(s,series,categories=short,{top=195,height=395,width=1140,left=70,max=1.05,legend=true}={}){
 series=series.map(v=>({...v,values:v.values.map(x=>Number(x.toFixed(6)))}));
 const ch=s.charts.add('bar',{position:{left,top,width,height},categories,series,barOptions:{direction:'column',grouping:'clustered',gapWidth:75},
 hasLegend:legend,legend:{position:'bottom',textStyle:{fontSize:21,fill:C.ink}},
 xAxis:{textStyle:{fontSize:21,fill:C.ink},line:{fill:C.line,width:1}},
 yAxis:{min:0,max,numberFormatCode:'0.0',majorUnit:.2,textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:C.line,width:1}},
 chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0},dataLabels:{showValue:false}});
 applyPresentationChartFont(ch,{fontFamily:'Arial'});return ch;
}
// 01
{
 const s=pres.slides.add();s.background.fill=C.blue;
 text(s,'Medical insurance\nclaim fraud detection',70,130,1100,205,68,'#FFFFFF',true);
 text(s,'Approach 1 and Approach 2',76,378,1050,58,34,'#FFFFFF');
 text(s,'Five classical models and three neural models',76,453,1050,55,27,'#DDE9E7');
 text(s,'20 October 2026',76,628,900,34,21,'#DDE9E7');
 s.speakerNotes.text='Research and demonstration based on the repository workbook. This deck covers the classical and neural comparison. It reports retrospective internal benchmarks and does not assert insurer deployment readiness.';
}
// 02
{
 const s=slide('Dataset and classification task','4,500 claims from the supplied workbook','Source: Health Insurance Fraud Claims.xlsx. The workbook origin and currency are undocumented.');
 text(s,'270',72,214,450,125,98,C.accent,true);text(s,'fraud labels',78,342,440,52,30);
 text(s,'6.0%',690,214,450,125,98,C.blue,true);text(s,'fraud prevalence',696,342,450,52,30);
 para(s,'An all-legitimate classifier reaches 94% accuracy while detecting no fraud.',456,1100,29);
 para(s,'No missing cells or duplicate complete rows. Dates span July 2022 to July 2024.',559,1100,24);
}
// 03
{
 const s=slide('Shared data partitions and feature contracts','Training transformations never fit on validation or test rows','Preprocessing follows training-only fitting. Source: https://scikit-learn.org/stable/common_pitfalls.html');
 table(s,['Partition','Claims','Fraud'],[['Training','3,150','189'],['Validation','675','41'],['Test','675','40']],{top:210,height:230,width:600,widths:[270,170,160]});
 text(s,'29 features\nOriginal benchmark',745,214,450,98,29,C.blue,true);
 text(s,'28 features\nIntake without Cluster',745,341,450,98,29,C.accent,true);
 para(s,'Every contract excludes ClaimStatus and identifiers. The amount ablation also removes ClaimAmount.',505,1100,28);
}
// 04
{
 const s=slide('Approach 1 classical algorithms','Five existing model families, retrained on the shared partitions');
 table(s,['Algorithm','Main configuration'],[
 ['Logistic Regression','Balanced weights, maximum 2,000 iterations'],['Random Forest','300 trees, balanced weights'],['XGBoost','100 trees, depth 6, learning rate 0.3'],['SVM','RBF kernel, balanced weights, sigmoid-margin score'],['Histogram GB','100 boosting iterations, balanced weights']],{height:360,widths:[340,804]});
 para(s,'Class weights use training data. The same encoded fields feed both approaches.',589,1110,24);
}
// 05
{
 const s=slide('Approach 2 neural algorithms','Existing MLP plus two implemented architectures','Source for weighted loss: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html. Residual MLP is a project implementation, not a reproduction of a published baseline.');
 table(s,['Algorithm','Architecture'],[['MLP','64 and 32 dense hidden units, 20% dropout'],['Residual MLP','64-unit projection and two residual blocks'],['1D CNN','Kernel-3 convolutions, 8 and 16 channels, dense 32']],{height:245,widths:[340,804]});
 para(s,'AdamW, learning rate 0.001, batch size 128 and weighted binary cross entropy.',493,1100,27);
 para(s,'Maximum 100 epochs. Patience 15. Best checkpoint by validation AP.',575,1100,25);
}
// 06
{
 const s=slide('Evaluation protocol','One defined procedure across all eight algorithms');
 para(s,'01  Fit preprocessing and model weights on training rows.',214,1110,30);
 para(s,'02  Select neural checkpoints by validation average precision.',307,1110,30);
 para(s,'03  Choose a decision threshold that maximizes validation F1.',400,1110,30);
 para(s,'04  Freeze the model and threshold before reporting test metrics.',493,1110,30);
 para(s,'Model ranking uses validation AP, then validation F1 and declared model order.',600,1100,22);
}
// 07
{
 const s=slide('Evaluation formulas','Fraud is the positive class','Sources: https://scikit-learn.org/stable/modules/model_evaluation.html and https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html. AP uses step weighting, not trapezoidal PR integration.');
 table(s,['Measure','Calculation'],[['Accuracy','(TP + TN) / all claims'],['Precision','TP / (TP + FP)'],['Recall','TP / (TP + FN)'],['F1','2 TP / (2 TP + FP + FN)'],['Average precision','Sum of recall increments × corresponding precision']],{height:355,widths:[320,824]});
 para(s,'AP and ROC AUC evaluate ranking. Precision, recall and F1 depend on the threshold.',588,1100,25);
}
// 08
{
 const s=slide('Original and intake model ranking','Test average precision on the same random partitions');
 chart(s,[{name:'Original 29 features',values:names.map(n=>get('baseline',n).ap),fill:'#8BA5A0'},{name:'Intake without Cluster',values:names.map(n=>get('intake',n).ap),fill:C.accent}]);
 para(s,'Removing Cluster preserves strong tree performance. Other model families lose ranking quality.',603,1110,23);
}
// 09
{
 const s=slide('Approach 1 intake test results','675 claims, including 40 fraud labels. Each model uses its validation threshold.');
 table(s,['Model','AP','Precision','Recall','F1','FP / FN'],names.slice(0,5).map((n,i)=>{const r=get('intake',n);return[labels[i],F(r.ap),P(r.precision),P(r.recall),F(r.f1),`${r.fp} / ${r.fn}`];}),{height:365,widths:[310,150,175,165,160,184]});
 para(s,'Random Forest is the selected default under the defined validation ranking rule.',602,1110,24);
}
// 10
{
 const s=slide('Approach 2 intake test results','The CNN has the highest neural AP. Threshold choice determines its recall.');
 table(s,['Model','AP','Precision','Recall','F1','FP / FN'],names.slice(5).map((n,i)=>{const r=get('intake',n);return[labels[i+5],F(r.ap),P(r.precision),P(r.recall),F(r.f1),`${r.fp} / ${r.fn}`];}),{height:265,widths:[310,150,175,165,160,184]});
 para(s,'The CNN follows the fixed encoded feature order, which has no inherent spatial meaning.',510,1110,28);
 para(s,'Greater architecture complexity does not establish a better operating point.',600,1110,23);
}
// 11
{
 const s=slide('Random Forest operating point','Intake contract. Threshold selected on validation: 0.4167.');
 table(s,['Actual class','Below threshold','Flagged'],[['Legitimate','635','0'],['Fraud','1','39']],{top:230,height:230,width:710,widths:[250,230,230]});
 text(s,'100.0% precision\n97.5% recall\n0.9873 F1',845,242,350,200,35,C.blue,true);
 para(s,'Accuracy = 674 / 675 = 99.8519%. Perfect AP can coexist with a missed fraud case.',536,1110,28);
}
// 12
{
 const s=slide('Amount sensitivity changes the conclusion','Test AP after removing Cluster and then claim amount');
 table(s,['Model','Original','No Cluster','No Cluster or amount'],names.map((n,i)=>[labels[i],F(get('baseline',n).ap),F(get('intake',n).ap),F(get('no_amount',n).ap)]),{height:395,widths:[330,210,245,359]});
 para(s,'Random Forest AP falls from 1.0000 to 0.2066 without amount. The dataset has a strong shortcut risk.',615,1120,23);
}
// 13
{
 const s=slide('Chronological evaluation','Training ends 21 November 2023. Validation ends 10 March 2024.');
 chart(s,[{name:'Chronological test F1',values:names.map(n=>get('temporal',n).f1),fill:C.blue}],short,{legend:false});
 para(s,'Later test period: 671 claims, including 37 fraud cases. Tree models have no errors in this internal split.',605,1110,23);
}
// 14
{
 const s=slide('Decision threshold sensitivity','Intake test F1 at the fixed and validation-selected policies');
 chart(s,[{name:'Threshold 0.5',values:names.map(n=>get('intake',n,'default').f1),fill:'#8BA5A0'},{name:'Validation-selected threshold',values:names.map(n=>get('intake',n).f1),fill:C.accent}]);
 para(s,'Validation tuning can help or hurt on test data. The test outcome does not choose the policy.',605,1110,23);
}
// 15
{
 const s=slide('Uncertainty and dataset limits','Strong internal results require careful interpretation');
 para(s,'Random Forest recall interval: 92.5% to 100.0%.\nF1 interval: 0.9610 to 1.0000.',210,1110,31);
 para(s,'The 1,000-resample bootstrap fixes the fitted model and threshold. It omits training and threshold-selection variability.',348,1110,28);
 para(s,'Data origin and label creation are undocumented. The random test split has appeared in earlier analyses.',468,1110,28);
 para(s,'Amount dependence and a degenerate AP interval prevent claims of external certainty.',585,1110,24);
}
// 16
{
 const s=slide('Claim Fraud Lab website','Both approaches share one working prediction interface');
 const bytes=await fs.readFile(path.join(OUT,'figures/website_demo.png'));
 s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:'Claim Fraud Lab website overview',fit:'contain',position:{left:65,top:196,width:760,height:429}});
 text(s,'Select either approach\n\nTest a claim or compare all 8\n\nSwitch experiment results\n\nDownload prediction JSON',870,225,340,330,27);
 text(s,'Inference runs locally in the browser from exported fitted parameters.',870,575,330,80,22,C.muted);
}
// 17
{
 const s=slide('Demonstration and consistency checks','A local Python API independently evaluates the saved artifacts');
 para(s,'1,350 preprocessing comparisons match exactly.',211,1110,34);
 para(s,'10,800 browser and Python scores match within 0.000001.\nNo threshold decision mismatches.',310,1110,31);
 para(s,'128 metric rows reconcile with independent formulas.\n32 API prediction requests pass.',449,1110,30);
 para(s,'Demo: load examples, test models, compare all eight, then inspect amount sensitivity.',591,1110,23);
}
// 18
{
 const s=slide('Conclusions','A reproducible comparison and model-testing demonstration','References: project workbook and submission/results artifacts; scikit-learn AP, model evaluation and common pitfalls documentation; PyTorch BCEWithLogitsLoss; Gorishniy et al., Revisiting Deep Learning Models for Tabular Data, NeurIPS 2021. https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html');
 para(s,'Tree models lead the internal benchmark.\nThe intake Random Forest detects 39 of 40 fraud cases with no false alerts.',216,1110,33);
 para(s,'Removing amount exposes substantial performance loss.\nIndependent claims and documented labels are the next research priority.',385,1110,31);
 para(s,'Live site: claim-fraud-lab-ravi-2026.clearbell16.chatgpt.site',574,1110,23);
}
await fs.mkdir(path.join(OUT,'.build/slides'),{recursive:true});
for(const [i,s]of pres.slides.items.entries()){
 const b=await pres.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(OUT,'.build/slides',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
const candidate=path.join(OUT,'.build/candidate.pptx');await(await PresentationFile.exportPptx(pres)).save(candidate);
const result=await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath:process.env.PRESENTATION_OUTPUT||path.join(OUT,'deliverables/Presentation.pptx'),
 pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[3,4,5,7,9,10,11,12].flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'design',families:['Arial']},
 requiredNativeTableOwnerSlides:[3,4,5,7,9,10,11,12],requiredNativeChartOwnerSlides:[8,13,14],
 materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,receiptPath:path.join(OUT,'.build/presentation.validation.json')});
console.log(JSON.stringify(result));

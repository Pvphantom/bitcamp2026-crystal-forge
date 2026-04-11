import { useState, useEffect, useCallback, useRef } from "react";

// ============================================================
// SIMULATION ENGINE (verified N=6 Schwinger model, 64-dim)
// ============================================================
const PAULI={I:[1,0,0,1],X:[0,1,1,0],Z:[1,0,0,-1]};
function buildRealOp(N,ops){const dim=1<<N;let r=PAULI[ops[N-1]].slice();let cd=2;for(let q=N-2;q>=0;q--){const p=PAULI[ops[q]];const nd=cd*2;const nr=new Float64Array(nd*nd);for(let i1=0;i1<cd;i1++)for(let j1=0;j1<cd;j1++)for(let i2=0;i2<2;i2++)for(let j2=0;j2<2;j2++)nr[(i1*2+i2)*nd+(j1*2+j2)]=r[i1*cd+j1]*p[i2*2+j2];r=nr;cd=nd;}return r;}
function buildYYop(N,q1,q2){const dim=1<<N;const m=new Float64Array(dim*dim);for(let i=0;i<dim;i++)for(let j=0;j<dim;j++){let ok=true;for(let q=0;q<N;q++){const bi=(i>>q)&1,bj=(j>>q)&1;if(q===q1||q===q2){if(bi===bj){ok=false;break;}}else{if(bi!==bj){ok=false;break;}}}if(!ok)continue;let re=1,im=0;for(const q of[q1,q2]){const fi=((i>>q)&1)===0?-1:1;const nr=-im*fi;const ni=re*fi;re=nr;im=ni;}m[i*dim+j]=re;}return m;}
function buildH(N,mass,g,eps0){const dim=1<<N;const H=new Float64Array(dim*dim);for(let n=0;n<N-1;n++){const ops=Array(N).fill('I');ops[n]='X';ops[n+1]='X';const xx=buildRealOp(N,ops);const yy=buildYYop(N,n,n+1);for(let i=0;i<dim*dim;i++)H[i]+=0.25*(xx[i]+yy[i]);}for(let nm=1;nm<=N;nm++){const sign=nm%2===0?1:-1;const ops=Array(N).fill('I');ops[nm-1]='Z';const zz=buildRealOp(N,ops);for(let i=0;i<dim*dim;i++)H[i]+=sign*mass/2*zz[i];}for(let nm=1;nm<N;nm++){let cn=eps0;for(let k=1;k<=nm;k++)cn-=0.5*Math.pow(-1,k);const Ln=new Float64Array(dim*dim);for(let i=0;i<dim;i++)Ln[i*dim+i]=cn;for(let k=1;k<=nm;k++){const ops=Array(N).fill('I');ops[k-1]='Z';const zk=buildRealOp(N,ops);for(let i=0;i<dim*dim;i++)Ln[i]-=0.5*zk[i];}for(let i=0;i<dim;i++)for(let j=0;j<dim;j++){let s=0;for(let k=0;k<dim;k++)s+=Ln[i*dim+k]*Ln[k*dim+j];H[i*dim+j]+=g*g/2*s;}}return{data:H,dim};}
function buildEvo(H,dt){const d=H.dim,n2=d*d,Hd=H.data;const Ur=new Float64Array(n2),Ui=new Float64Array(n2);for(let i=0;i<d;i++)Ur[i*d+i]=1;let Tr=new Float64Array(n2),Ti=new Float64Array(n2);for(let i=0;i<d;i++)Tr[i*d+i]=1;for(let k=1;k<=30;k++){const nTr=new Float64Array(n2),nTi=new Float64Array(n2);for(let i=0;i<d;i++)for(let j=0;j<d;j++){let sr=0,si=0;for(let m=0;m<d;m++){const aim=-Hd[m*d+j]*dt;sr+=-Ti[i*d+m]*aim;si+=Tr[i*d+m]*aim;}nTr[i*d+j]=sr/k;nTi[i*d+j]=si/k;}Tr=nTr;Ti=nTi;for(let i=0;i<n2;i++){Ur[i]+=Tr[i];Ui[i]+=Ti[i];}}return{re:Ur,im:Ui,dim:d};}
function applyU(U,vr,vi){const d=U.dim;const or=new Float64Array(d),oi=new Float64Array(d);for(let i=0;i<d;i++){let sr=0,si=0;for(let j=0;j<d;j++){sr+=U.re[i*d+j]*vr[j]-U.im[i*d+j]*vi[j];si+=U.re[i*d+j]*vi[j]+U.im[i*d+j]*vr[j];}or[i]=sr;oi[i]=si;}return{re:or,im:oi};}
function makeState(N,p,q){const dim=1<<N;let idx=0;for(let nm=2;nm<=N;nm+=2)idx|=(1<<(nm-1));if(p>0&&q>0){idx^=(1<<(p-1));idx^=(1<<(q-1));}return{re:Float64Array.from({length:dim},(_,i)=>i===idx?1:0),im:new Float64Array(dim)};}
function getObs(re,im,N,mass,g,eps0,H){const dim=1<<N;const Z=[];for(let q=0;q<N;q++){let v=0;for(let i=0;i<dim;i++){const p=re[i]*re[i]+im[i]*im[i];v+=p*(((i>>q)&1)===0?1:-1);}Z.push(v);}const L=[];for(let nm=1;nm<N;nm++){let v=eps0;for(let k=1;k<=nm;k++)v-=0.5*(Z[k-1]+Math.pow(-1,k));L.push(v);}const Q=[];for(let nm=1;nm<=N;nm++)Q.push(-(Z[nm-1]+Math.pow(-1,nm))/2);const occ=Z.map(z=>(1-z)/2);let energy=0;for(let i=0;i<dim;i++){let hr=0;for(let j=0;j<dim;j++)hr+=H.data[i*dim+j]*re[j];let hi=0;for(let j=0;j<dim;j++)hi+=H.data[i*dim+j]*im[j];energy+=re[i]*hr+im[i]*hi;}let ch=0;for(let nm=1;nm<=N;nm++)ch+=Math.pow(-1,nm)*Z[nm-1];ch*=-1/(2*N);return{Z,L,Q,occupation:occ,energy,chiral:ch};}
function eigenMin(H){const d=H.dim,Hd=H.data;let shift=0;for(let i=0;i<d;i++)shift=Math.max(shift,Math.abs(Hd[i*d+i]));shift*=d;let v=new Float64Array(d);for(let i=0;i<d;i++)v[i]=Math.sin(i*7.3+1.2);let norm=Math.sqrt(v.reduce((s,x)=>s+x*x,0));for(let i=0;i<d;i++)v[i]/=norm;for(let iter=0;iter<400;iter++){const w=new Float64Array(d);for(let i=0;i<d;i++){let s=shift*v[i];for(let j=0;j<d;j++)s-=Hd[i*d+j]*v[j];w[i]=s;}norm=Math.sqrt(w.reduce((s,x)=>s+x*x,0));for(let i=0;i<d;i++)v[i]=w[i]/norm;}let num=0;for(let i=0;i<d;i++){let s=0;for(let j=0;j<d;j++)s+=Hd[i*d+j]*v[j];num+=v[i]*s;}return num;}

// ============================================================
// LEVELS
// ============================================================
const LEVELS = [
  {
    id: 1, name: "Tutorial", subtitle: "Learn the basics",
    desc: "The particles are close together and the string is weak. Just hit Play and watch it snap on its own.",
    p: 1, q: 4, m: 0.1, g: 0.5, eps0: 0,
    unlocked: [], // no sliders — just watch
    timeLimit: 30, breakThreshold: 40,
    stars: [25, 15, 8], // seconds for 1, 2, 3 stars
    hint: "This one breaks by itself — just press Play!",
  },
  {
    id: 2, name: "Lightweight", subtitle: "Use the mass slider",
    desc: "The string is stronger now. Make the particles lighter so new ones can pop into existence more easily.",
    p: 1, q: 6, m: 1.5, g: 1.0, eps0: 0,
    unlocked: ['m'], // only mass slider
    timeLimit: 45, breakThreshold: 50,
    stars: [35, 20, 12],
    hint: "Lighter particles are easier to create from vacuum energy. Try dragging mass toward zero.",
  },
  {
    id: 3, name: "Weakener", subtitle: "Use the string strength slider",
    desc: "Heavy particles, strong string. Weaken the string so it can't hold them together anymore.",
    p: 1, q: 6, m: 0.8, g: 2.5, eps0: 0,
    unlocked: ['g'], // only coupling
    timeLimit: 45, breakThreshold: 50,
    stars: [35, 22, 14],
    hint: "A weaker string stores less energy — at some point it can't hold the particles.",
  },
  {
    id: 4, name: "Dual Wield", subtitle: "Two sliders, tighter deadline",
    desc: "You've got both mass and string strength. Find the right combo to snap the string before time runs out.",
    p: 1, q: 6, m: 1.2, g: 2.0, eps0: 0,
    unlocked: ['m', 'g'],
    timeLimit: 35, breakThreshold: 55,
    stars: [28, 18, 10],
    hint: "Light particles + weak string = fast breaking. But how low do you need to go?",
  },
  {
    id: 5, name: "Background Noise", subtitle: "All three controls",
    desc: "A background field is pushing through the system. Use everything you've got.",
    p: 1, q: 6, m: 1.0, g: 2.0, eps0: 0.3,
    unlocked: ['m', 'g', 'eps0'],
    timeLimit: 40, breakThreshold: 55,
    stars: [32, 20, 12],
    hint: "The background field shifts the vacuum. Sometimes that helps, sometimes it hurts.",
  },
  {
    id: 6, name: "Speed Run", subtitle: "Break it in under 8 seconds",
    desc: "You know all the tricks now. Parameters are locked at hard values. Find the fastest path to breaking.",
    p: 1, q: 6, m: 1.5, g: 2.5, eps0: 0,
    unlocked: ['m', 'g', 'eps0'],
    timeLimit: 25, breakThreshold: 55,
    stars: [20, 12, 8],
    hint: "Go extreme: mass near zero, coupling low, then let it rip.",
  },
];

// ============================================================
// UI
// ============================================================
const C = {
  bg:'#0a0b12', card:'#12141e', cardHi:'#181b28', border:'#232840',
  blue:'#4d8bff', blueG:'rgba(77,139,255,0.35)',
  red:'#ff3d6a', redG:'rgba(255,61,106,0.35)',
  gold:'#ffbf00', goldDim:'rgba(255,191,0,0.15)',
  purple:'#b266ff', teal:'#00e5c3',
  text:'#e0e4ef', soft:'#8891ab', faint:'#4a5170',
  green:'#00e676', warn:'#ffa726',
};

function Star({ filled }) {
  return <span style={{ fontSize: 18, color: filled ? C.gold : C.faint }}>{filled ? '★' : '☆'}</span>;
}

export default function App() {
  const N = 6;
  // Game state
  const [mode, setMode] = useState('menu'); // menu | playing | won | lost | sandbox
  const [levelIdx, setLevelIdx] = useState(0);
  const [bestStars, setBestStars] = useState({});
  const [timer, setTimer] = useState(0);
  const [showHint, setShowHint] = useState(false);

  // Sim state
  const [params, setParams] = useState({ m: 0.5, g: 1.0, eps0: 0 });
  const [sites, setSites] = useState({ p: 1, q: 6 });
  const [psi, setPsi] = useState(null);
  const [obs, setObs] = useState(null);
  const [ham, setHam] = useState(null);
  const [evo, setEvo] = useState(null);
  const [e0, setE0] = useState(null);
  const [steps, setSteps] = useState(0);
  const [running, setRunning] = useState(false);
  const [ready, setReady] = useState(false);
  const [peakBreak, setPeakBreak] = useState(0);

  const timerRef = useRef(null);
  const clockRef = useRef(null);
  const evoRef = useRef(null);
  const hamRef = useRef(null);
  const parRef = useRef(params);

  const level = LEVELS[levelIdx];

  // Init simulation
  const initSim = useCallback((pp, ss) => {
    setReady(false); setRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);
    if (clockRef.current) clearInterval(clockRef.current);
    setTimeout(() => {
      const H = buildH(N, pp.m, pp.g, pp.eps0);
      const U = buildEvo(H, 0.2);
      setHam(H); hamRef.current = H;
      setEvo(U); evoRef.current = U;
      parRef.current = pp;
      setE0(eigenMin(H));
      const s = makeState(N, ss.p, ss.q);
      setPsi(s);
      setObs(getObs(s.re, s.im, N, pp.m, pp.g, pp.eps0, H));
      setSteps(0); setPeakBreak(0); setReady(true);
    }, 20);
  }, []);

  // Start a level
  const startLevel = (idx) => {
    const lv = LEVELS[idx];
    setLevelIdx(idx);
    const pp = { m: lv.m, g: lv.g, eps0: lv.eps0 };
    setParams(pp);
    setSites({ p: lv.p, q: lv.q });
    setMode('playing');
    setTimer(0);
    setShowHint(false);
    initSim(pp, { p: lv.p, q: lv.q });
  };

  const startSandbox = () => {
    const pp = { m: 0.5, g: 1.0, eps0: 0 };
    setParams(pp);
    setSites({ p: 1, q: 6 });
    setMode('sandbox');
    setTimer(0);
    initSim(pp, { p: 1, q: 6 });
  };

  // Evolution step
  const doStep = useCallback(() => {
    setPsi(prev => {
      if (!prev || !evoRef.current) return prev;
      const next = applyU(evoRef.current, prev.re, prev.im);
      const p = parRef.current;
      const o = getObs(next.re, next.im, N, p.m, p.g, p.eps0, hamRef.current);
      setObs(o);
      setSteps(s => s + 1);
      // Track peak break
      const innerQ = o.Q.slice(1, -1).reduce((s, q) => s + Math.abs(q), 0);
      const bp = Math.min(100, Math.round(innerQ / (N - 2) * 100));
      setPeakBreak(prev => Math.max(prev, bp));
      return next;
    });
  }, []);

  const toggleAuto = () => {
    if (running) {
      clearInterval(timerRef.current);
      setRunning(false);
    } else {
      setRunning(true);
      timerRef.current = setInterval(doStep, 120);
    }
  };

  // Game clock
  useEffect(() => {
    if (mode === 'playing' && ready) {
      clockRef.current = setInterval(() => setTimer(t => t + 1), 1000);
      return () => clearInterval(clockRef.current);
    }
    return () => {};
  }, [mode, ready]);

  // Check win/lose
  useEffect(() => {
    if (mode !== 'playing' || !obs) return;
    const innerQ = obs.Q.slice(1, -1).reduce((s, q) => s + Math.abs(q), 0);
    const bp = Math.min(100, Math.round(innerQ / (N - 2) * 100));
    if (bp >= level.breakThreshold) {
      // WIN
      setRunning(false);
      if (timerRef.current) clearInterval(timerRef.current);
      if (clockRef.current) clearInterval(clockRef.current);
      const earnedStars = timer <= level.stars[2] ? 3 : timer <= level.stars[1] ? 2 : timer <= level.stars[0] ? 1 : 1;
      setBestStars(prev => ({
        ...prev,
        [level.id]: Math.max(prev[level.id] || 0, earnedStars),
      }));
      setMode('won');
    } else if (timer >= level.timeLimit) {
      setRunning(false);
      if (timerRef.current) clearInterval(timerRef.current);
      if (clockRef.current) clearInterval(clockRef.current);
      setMode('lost');
    }
  }, [obs, timer, mode]);

  useEffect(() => () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (clockRef.current) clearInterval(clockRef.current);
  }, []);

  const changeParam = (k, v) => {
    const pp = { ...params, [k]: v }; setParams(pp);
    if (running) { clearInterval(timerRef.current); setRunning(false); }
    const H = buildH(N, pp.m, pp.g, pp.eps0);
    const U = buildEvo(H, 0.2);
    setHam(H); hamRef.current = H; setEvo(U); evoRef.current = U; parRef.current = pp;
    setE0(eigenMin(H));
    if (psi) setObs(getObs(psi.re, psi.im, N, pp.m, pp.g, pp.eps0, H));
  };

  const clickSite = (nm) => {
    if (mode === 'playing') return; // no dragging in challenge
    const isOdd = nm % 2 === 1;
    if (isOdd && nm !== sites.p) { const ss = { p: nm, q: sites.q }; setSites(ss); initSim(params, ss); }
    else if (!isOdd && nm !== sites.q) { const ss = { p: sites.p, q: nm }; setSites(ss); initSim(params, ss); }
  };

  // Computed
  const innerQ = obs ? obs.Q.slice(1, -1).reduce((s, q) => s + Math.abs(q), 0) : 0;
  const breakPct = Math.min(100, Math.round(innerQ / (N - 2) * 100));
  const isBroken = breakPct >= (level?.breakThreshold || 60);
  const isBreaking = breakPct > 15;
  const maxL = obs ? Math.max(0.15, ...obs.L.map(Math.abs)) : 1;
  const eR = Math.max(2, Math.abs(e0 || 0) * 1.5, Math.abs(obs?.energy || 0) * 1.5);
  const sp = 520 / (N + 1);
  const isPlaying = mode === 'playing' || mode === 'sandbox';
  const timeLeft = level ? level.timeLimit - timer : 0;

  const sliderDefs = [
    { k: 'g', label: 'String Strength', desc: 'Lower = easier to break', min: 0.1, max: 3, step: 0.05 },
    { k: 'm', label: 'Particle Weight', desc: 'Lower = easier to create from nothing', min: 0, max: 2, step: 0.05 },
    { k: 'eps0', label: 'Background Field', desc: 'A constant force through the system', min: 0, max: 1, step: 0.02 },
  ];

  // ============================================================
  // MENU SCREEN
  // ============================================================
  if (mode === 'menu') {
    return (
      <div style={{ background: C.bg, minHeight: '100vh', color: C.text, fontFamily: "'Outfit','Segoe UI',sans-serif", padding: '20px 12px' }}>
        <div style={{ maxWidth: 500, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 24 }}>
            <div style={{ fontSize: 42, marginBottom: 4 }}>⚛️</div>
            <h1 style={{
              fontSize: 32, fontWeight: 800, margin: 0,
              background: `linear-gradient(135deg,${C.blue},${C.teal},${C.red})`,
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            }}>String Breaker</h1>
            <p style={{ fontSize: 13, color: C.soft, margin: '8px 0 0', lineHeight: 1.5 }}>
              Pull particles apart until the energy string between them snaps —<br />
              and new particles are born from empty space.
            </p>
            <p style={{ fontSize: 10, color: C.faint, margin: '6px 0 0' }}>
              A real quantum simulation of quark confinement
            </p>
          </div>

          {/* Levels */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, color: C.teal, fontWeight: 700, letterSpacing: '0.1em', marginBottom: 8 }}>CHALLENGE MODE</div>
            {LEVELS.map((lv, i) => {
              const stars = bestStars[lv.id] || 0;
              const locked = i > 0 && !bestStars[LEVELS[i - 1].id];
              return (
                <button key={lv.id} onClick={() => !locked && startLevel(i)} style={{
                  width: '100%', display: 'flex', alignItems: 'center', gap: 12,
                  padding: '12px 14px', marginBottom: 6,
                  background: locked ? C.card : C.cardHi, border: `1px solid ${locked ? C.faint : C.border}`,
                  borderRadius: 10, cursor: locked ? 'not-allowed' : 'pointer',
                  opacity: locked ? 0.4 : 1, textAlign: 'left', color: C.text,
                  fontFamily: 'inherit', transition: 'background 0.15s',
                }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: 8,
                    background: locked ? C.faint : `linear-gradient(135deg,${C.blue},${C.teal})`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 16, fontWeight: 800, color: '#fff', flexShrink: 0,
                  }}>{locked ? '🔒' : lv.id}</div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 700 }}>{lv.name}</div>
                    <div style={{ fontSize: 10, color: C.soft }}>{lv.subtitle}</div>
                  </div>
                  <div style={{ display: 'flex', gap: 1, flexShrink: 0 }}>
                    {[1, 2, 3].map(s => <Star key={s} filled={s <= stars} />)}
                  </div>
                  <div style={{ fontSize: 9, color: C.faint, width: 30, textAlign: 'right', flexShrink: 0 }}>
                    {lv.timeLimit}s
                  </div>
                </button>
              );
            })}
          </div>

          {/* Sandbox */}
          <button onClick={startSandbox} style={{
            width: '100%', padding: '14px', background: 'transparent',
            border: `1px dashed ${C.soft}`, borderRadius: 10, cursor: 'pointer',
            color: C.soft, fontSize: 12, fontFamily: 'inherit', fontWeight: 600,
          }}>
            🔬 Free Play — no timer, all controls unlocked
          </button>
        </div>
      </div>
    );
  }

  // ============================================================
  // WIN / LOSE SCREENS
  // ============================================================
  if (mode === 'won' || mode === 'lost') {
    const won = mode === 'won';
    const earnedStars = won ? (timer <= level.stars[2] ? 3 : timer <= level.stars[1] ? 2 : 1) : 0;
    const nextIdx = levelIdx + 1;
    const hasNext = nextIdx < LEVELS.length;
    return (
      <div style={{
        background: C.bg, minHeight: '100vh', color: C.text,
        fontFamily: "'Outfit','Segoe UI',sans-serif", padding: '20px 12px',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <div style={{ textAlign: 'center', maxWidth: 400 }}>
          <div style={{ fontSize: 56, marginBottom: 8 }}>{won ? '🎉' : '⏰'}</div>
          <h2 style={{
            fontSize: 28, fontWeight: 800, margin: '0 0 4px',
            color: won ? C.green : C.red,
          }}>{won ? 'String Broken!' : 'Time\'s Up!'}</h2>
          <p style={{ fontSize: 13, color: C.soft, margin: '0 0 16px' }}>
            {won ? `You broke the string in ${timer} seconds` : `The string held for ${level.timeLimit} seconds`}
          </p>
          {won && (
            <div style={{ marginBottom: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'center', gap: 4, marginBottom: 4 }}>
                {[1, 2, 3].map(s => <Star key={s} filled={s <= earnedStars} />)}
              </div>
              <div style={{ fontSize: 10, color: C.faint }}>
                {earnedStars === 3 ? 'Perfect! ⚡' : earnedStars === 2 ? 'Great job!' : `Try under ${level.stars[1]}s for ★★`}
              </div>
              {level.stars[2] > 0 && earnedStars < 3 && (
                <div style={{ fontSize: 9, color: C.faint, marginTop: 2 }}>
                  ★★★ under {level.stars[2]}s
                </div>
              )}
            </div>
          )}
          <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
            <button onClick={() => startLevel(levelIdx)} style={{
              padding: '10px 24px', border: `1px solid ${C.border}`, background: C.cardHi,
              borderRadius: 8, color: C.text, fontSize: 12, cursor: 'pointer', fontFamily: 'inherit', fontWeight: 600,
            }}>↺ Retry</button>
            {won && hasNext && (
              <button onClick={() => startLevel(nextIdx)} style={{
                padding: '10px 24px', border: 'none',
                background: `linear-gradient(135deg,${C.teal},${C.blue})`,
                borderRadius: 8, color: '#fff', fontSize: 12, cursor: 'pointer', fontFamily: 'inherit', fontWeight: 700,
              }}>Next Level →</button>
            )}
            <button onClick={() => setMode('menu')} style={{
              padding: '10px 24px', border: `1px solid ${C.faint}`, background: 'transparent',
              borderRadius: 8, color: C.soft, fontSize: 12, cursor: 'pointer', fontFamily: 'inherit',
            }}>Menu</button>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================
  // GAME / SANDBOX SCREEN
  // ============================================================
  if (!obs || !ready) return (
    <div style={{ background: C.bg, minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: C.teal, fontFamily: "'Outfit',sans-serif", fontSize: 14 }}>
      <div style={{ textAlign: 'center' }}><div style={{ fontSize: 36, marginBottom: 8 }}>⚛️</div>Setting up quantum simulation...</div>
    </div>
  );

  return (
    <div style={{
      background: C.bg, minHeight: '100vh', color: C.text,
      fontFamily: "'Outfit','Segoe UI',sans-serif", padding: '10px 8px',
    }}>
      <div style={{ maxWidth: 660, margin: '0 auto' }}>

        {/* Top bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <button onClick={() => { setRunning(false); if (timerRef.current) clearInterval(timerRef.current); setMode('menu'); }} style={{
            background: 'none', border: 'none', color: C.soft, fontSize: 11, cursor: 'pointer', fontFamily: 'inherit',
          }}>← Back</button>

          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 14, fontWeight: 700 }}>
              {mode === 'sandbox' ? '🔬 Free Play' : `Level ${level.id}: ${level.name}`}
            </div>
            {mode === 'playing' && (
              <div style={{ fontSize: 10, color: C.soft }}>{level.subtitle}</div>
            )}
          </div>

          {mode === 'playing' ? (
            <div style={{
              fontSize: 18, fontWeight: 800, fontFamily: "'JetBrains Mono',monospace",
              color: timeLeft <= 10 ? C.red : timeLeft <= 20 ? C.warn : C.teal,
            }}>
              {timeLeft}s
            </div>
          ) : <div style={{ width: 40 }} />}
        </div>

        {/* Level description */}
        {mode === 'playing' && (
          <div style={{
            background: C.cardHi, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: '8px 12px', marginBottom: 8, fontSize: 11, color: C.soft, lineHeight: 1.5,
          }}>
            {level.desc}
            {!showHint && (
              <button onClick={() => setShowHint(true)} style={{
                background: 'none', border: 'none', color: C.gold, fontSize: 10,
                cursor: 'pointer', fontFamily: 'inherit', marginLeft: 8,
              }}>💡 Hint</button>
            )}
            {showHint && <div style={{ color: C.gold, marginTop: 4 }}>💡 {level.hint}</div>}
          </div>
        )}

        {/* Status */}
        <div style={{ textAlign: 'center', marginBottom: 6, minHeight: 24 }}>
          {isBroken ? (
            <span style={{ display: 'inline-block', padding: '3px 14px', borderRadius: 20, background: `${C.green}18`, border: `1px solid ${C.green}`, color: C.green, fontSize: 10, fontWeight: 700 }}>
              ⚡ String snapped! New particles from the vacuum!
            </span>
          ) : isBreaking ? (
            <span style={{ display: 'inline-block', padding: '3px 14px', borderRadius: 20, background: `${C.warn}18`, border: `1px solid ${C.warn}`, color: C.warn, fontSize: 10, fontWeight: 700 }}>
              String weakening... {breakPct}%
            </span>
          ) : steps > 0 ? (
            <span style={{ display: 'inline-block', padding: '3px 14px', borderRadius: 20, background: `${C.blue}18`, border: `1px solid ${C.blue}`, color: C.blue, fontSize: 10, fontWeight: 700 }}>
              Evolving... {breakPct}%
            </span>
          ) : null}
        </div>

        {/* LATTICE VIS */}
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: '14px 6px 6px', marginBottom: 8 }}>
          <svg width="100%" height="155" viewBox="0 0 540 155" preserveAspectRatio="xMidYMid meet">
            {obs.L.map((Ln, i) => {
              const x1 = sp * (i + 1), x2 = sp * (i + 2), xm = (x1 + x2) / 2;
              const abs = Math.abs(Ln), pct = abs / maxL;
              const th = Math.max(1, pct * 18);
              const col = Ln > 0.01 ? C.gold : Ln < -0.01 ? C.purple : C.faint;
              const op = Math.max(0.08, pct);
              const breaking = pct < 0.6 && abs > 0.01;
              return (
                <g key={`l${i}`}>
                  {abs > 0.03 && <line x1={x1} y1={70} x2={x2} y2={70} stroke={col} strokeWidth={th + 14} opacity={op * 0.06} strokeLinecap="round" />}
                  {abs > 0.03 && <line x1={x1} y1={70} x2={x2} y2={70} stroke={col} strokeWidth={th + 6} opacity={op * 0.15} strokeLinecap="round" />}
                  <line x1={x1} y1={70} x2={x2} y2={70} stroke={col} strokeWidth={th} opacity={op * 0.9} strokeLinecap="round" strokeDasharray={breaking ? '6 4' : 'none'} />
                  <text x={xm} y={52} textAnchor="middle" fill={col} fontSize="9" opacity={0.85} fontWeight="600">
                    {abs > 0.01 ? `${Math.round(abs * 100)}%` : ''}
                  </text>
                </g>
              );
            })}
            {Array.from({ length: N }, (_, i) => {
              const nm = i + 1, x = sp * (i + 1);
              const ch = obs.Q[i], isF = ch > 0.3, isA = ch < -0.3, has = isF || isA;
              const forming = Math.abs(ch) > 0.1 && Math.abs(ch) <= 0.3;
              const col = ch > 0.1 ? C.blue : ch < -0.1 ? C.red : C.faint;
              const glow = ch > 0.1 ? C.blueG : ch < -0.1 ? C.redG : 'transparent';
              const str = Math.abs(ch), r = has ? 17 : forming ? 13 : 11;
              return (
                <g key={`s${i}`} onClick={() => clickSite(nm)} style={{ cursor: mode === 'sandbox' ? 'pointer' : 'default' }}>
                  {(has || forming) && <circle cx={x} cy={70} r={r + 14} fill={glow} opacity={has ? 0.4 : 0.15}>
                    {has && <animate attributeName="r" values={`${r + 8};${r + 18};${r + 8}`} dur="2s" repeatCount="indefinite" />}
                  </circle>}
                  <circle cx={x} cy={70} r={r}
                    fill={str > 0.1 ? col : 'transparent'} fillOpacity={has ? 1 : str * 3}
                    stroke={nm % 2 === 1 ? C.blue : C.red} strokeWidth={has ? 0 : 1}
                    strokeDasharray={has ? 'none' : '5 4'} opacity={has ? 1 : forming ? 0.7 : 0.2}
                  />
                  <text x={x} y={has ? 76 : 74} textAnchor="middle" fill="#fff" fontSize={has ? 18 : forming ? 12 : 0} fontWeight="bold" opacity={has ? 1 : str * 2.5}>
                    {isF ? '+' : isA ? '−' : forming ? (ch > 0 ? '+' : '−') : ''}
                  </text>
                  <text x={x} y={100} textAnchor="middle" fill={C.soft} fontSize="7">{nm % 2 === 1 ? 'matter' : 'anti'}</text>
                  {str > 0.02 && <g>
                    <rect x={x - 12} y={110} width={24} height={3} rx={1.5} fill={C.faint} opacity={0.3} />
                    <rect x={x - 12} y={110} width={Math.max(1, str * 24)} height={3} rx={1.5} fill={col} opacity={0.7} />
                  </g>}
                </g>
              );
            })}
            <circle cx={36} cy={140} r={4} fill={C.blue} /><text x={46} y={143} fill={C.soft} fontSize="7">matter</text>
            <circle cx={100} cy={140} r={4} fill={C.red} /><text x={110} y={143} fill={C.soft} fontSize="7">antimatter</text>
            <line x1={175} y1={140} x2={200} y2={140} stroke={C.gold} strokeWidth={3} /><text x={208} y={143} fill={C.soft} fontSize="7">energy string</text>
          </svg>
        </div>

        {/* Controls row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {/* Left: Play controls */}
          <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 12 }}>
            <button onClick={toggleAuto} style={{
              width: '100%', padding: '11px 0', border: 'none', borderRadius: 8,
              background: running ? `linear-gradient(135deg,${C.red},#8f2a42)` : `linear-gradient(135deg,${C.green},${C.teal})`,
              color: '#fff', fontSize: 13, fontWeight: 700, cursor: 'pointer',
              fontFamily: 'inherit', marginBottom: 6,
              boxShadow: running ? `0 3px 16px ${C.redG}` : `0 3px 16px rgba(0,230,118,0.15)`,
            }}>{running ? '⏸ Pause' : '▶ Play'}</button>
            <div style={{ display: 'flex', gap: 5 }}>
              <button onClick={doStep} style={{
                flex: 1, padding: '7px 0', border: `1px solid ${C.border}`, background: C.cardHi,
                color: C.soft, borderRadius: 6, cursor: 'pointer', fontSize: 9, fontFamily: 'inherit',
              }}>Step →</button>
              <button onClick={() => mode === 'sandbox' ? initSim(params, sites) : startLevel(levelIdx)} style={{
                flex: 1, padding: '7px 0', border: `1px solid ${C.border}`, background: C.cardHi,
                color: C.soft, borderRadius: 6, cursor: 'pointer', fontSize: 9, fontFamily: 'inherit',
              }}>↺ Reset</button>
            </div>
            {/* Energy */}
            <div style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ fontSize: 9, color: C.soft }}>System Energy</span>
                <span style={{ fontSize: 10, fontWeight: 700, color: obs.energy > 0 ? C.gold : C.teal }}>{obs.energy.toFixed(2)}</span>
              </div>
              <div style={{ position: 'relative', height: 8, background: C.cardHi, borderRadius: 4, overflow: 'hidden' }}>
                <div style={{
                  position: 'absolute', left: obs.energy < 0 ? `${50 + (obs.energy / eR) * 50}%` : '50%',
                  width: `${Math.abs(obs.energy) / eR * 50}%`, top: 0, bottom: 0,
                  background: obs.energy < 0 ? `${C.teal}88` : `${C.gold}88`, borderRadius: 4,
                }} />
                {e0 !== null && <div style={{ position: 'absolute', left: `${50 + (e0 / eR) * 50}%`, top: 0, bottom: 0, width: 2, background: C.green }} />}
              </div>
            </div>
            {/* Break progress */}
            <div style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ fontSize: 9, color: C.soft }}>Break progress</span>
                <span style={{ fontSize: 10, fontWeight: 700, color: isBroken ? C.green : isBreaking ? C.warn : C.soft }}>{breakPct}%</span>
              </div>
              <div style={{ position: 'relative', height: 8, background: C.cardHi, borderRadius: 4, overflow: 'hidden' }}>
                <div style={{
                  width: `${breakPct}%`, height: '100%', borderRadius: 4,
                  background: isBroken ? C.green : `linear-gradient(90deg,${C.blue},${C.teal})`,
                  transition: 'width 0.12s',
                }} />
                {mode === 'playing' && <div style={{
                  position: 'absolute', left: `${level.breakThreshold}%`, top: -2, bottom: -2,
                  width: 2, background: C.warn, borderRadius: 1,
                }} />}
              </div>
              {mode === 'playing' && <div style={{ fontSize: 7, color: C.faint, marginTop: 1 }}>
                Target: {level.breakThreshold}%
              </div>}
            </div>
          </div>

          {/* Right: Parameters */}
          <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: 12 }}>
            <div style={{ fontSize: 10, color: C.gold, fontWeight: 700, marginBottom: 8 }}>🎛 Parameters</div>
            {sliderDefs.map(({ k, label, desc, min, max, step }) => {
              const locked = mode === 'playing' && !level.unlocked.includes(k);
              return (
                <div key={k} style={{ marginBottom: 10, opacity: locked ? 0.35 : 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
                    <span style={{ fontSize: 10, fontWeight: 600, color: C.text }}>
                      {locked && '🔒 '}{label}
                    </span>
                    <span style={{ fontSize: 10, fontWeight: 700, color: C.gold, fontFamily: "'JetBrains Mono',monospace" }}>
                      {params[k].toFixed(2)}
                    </span>
                  </div>
                  <div style={{ fontSize: 7, color: C.faint, marginBottom: 2 }}>{desc}</div>
                  <input type="range" min={min} max={max} step={step} value={params[k]}
                    disabled={locked}
                    onChange={e => changeParam(k, parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: C.gold }} />
                </div>
              );
            })}
          </div>
        </div>

      </div>
    </div>
  );
}

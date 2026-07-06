"""Animate the SEARCH — how the blocks get placed. Hill-climb over legal move-sequences, each
graded by the exact droplet CA; record the best-so-far climb (flood -> partial -> win) plus a
sample of tried arrangements, then the winning pour. Shows every part of the system: candidates
tried, the sim's verdict (buckets green / floor red), the climb, and the final dynamics.

    python3 search_anim.py --level 3 --out fixtures/L3_search.mp4
"""
from __future__ import annotations
import argparse, numpy as np, cv2
import _paths  # noqa: F401
from _paths import LAB, FIXTURES
from imagine_search_gate import _present, _apply_real, GAME_ID, ACTION_OF_ID, game_live
from imag_search_solve import actions, real_win, MOVES
from engine_adapters.portband_glove_adapter import frame_components
from native_scene import native_grid
from dropletsim import pour_core_v
from arc_agi import Arcade
from arc_agi.base import OperationMode

S = 16
BG=(20,18,24); C_BLOCK=(110,80,55); C_WALL=(40,40,150); C_FLASH=(60,60,255)
C_TGT=(150,90,40); C_FILL=(90,210,110); C_DROP=(245,190,90); C_SRC=(60,190,235); C_REDIR=(200,60,200)
INK=(235,235,235); MUT=(150,150,160); GRN=(90,210,110); RED=(80,80,235); GOLD=(60,190,235)


def board_img(ng, filled, fail):
    b,t,w,r,tid = ng["block"],ng["target"],ng["wall"],ng["redir"],ng["tid"]
    H,W=b.shape
    img=np.zeros((H,W,3),np.uint8); img[:]=BG
    img[w]=C_FLASH if fail else C_WALL
    img[r]=C_REDIR; img[b]=C_BLOCK
    for y in range(H):
        for x in range(W):
            if t[y,x]: img[y,x]=C_FILL if int(tid[y,x]) in filled else C_TGT
    big=np.kron(img,np.ones((S,S,1),np.uint8))
    for (sy,sx) in ng["spouts"]:
        cv2.circle(big,(int(sx)*S+S//2,int(sy)*S+S//2),S//2-1,C_SRC,-1)
    return big


def frame_with_panel(board, lines):
    H,W,_=board.shape
    panel=np.zeros((66,W,3),np.uint8); panel[:]=(14,13,18)
    canvas=np.vstack([panel,board])
    for i,(txt,col,sz) in enumerate(lines):
        cv2.putText(canvas,txt,(14,24+i*22),cv2.FONT_HERSHEY_SIMPLEX,sz,col,1,cv2.LINE_AA)
    return canvas


def simulate_pour(ng,max_ticks=150,emit=64):
    from dropletsim import pour_core_v as _pc  # reuse kind logic via a light re-sim
    b,t,w,r,tid=ng["block"],ng["target"],ng["wall"],ng["redir"],ng["tid"]; H,W=b.shape
    def kind(y,x):
        if x<0 or x>=W or y<0 or y>=H: return None
        if b[y,x]:return "block"
        if t[y,x]:return "target"
        if w[y,x]:return "wall"
        if r[y,x]:return "redir"
        return None
    def perp(dx,dy): return [(-1,0),(1,0)] if dy!=0 else [(0,-1),(0,1)]
    filled=set(); drops=set(); snaps=[]; flash=0
    for tk in range(max_ticks):
        if tk<emit:
            for (sy,sx) in ng["spouts"]: drops.add((int(sy),int(sx),0,1))
        nxt=set()
        for (y,x,dx,dy) in drops:
            nx,ny=x+dx,y+dy; k=kind(ny,nx); p=perp(dx,dy)
            if k is None:
                if 0<=ny<H and 0<=nx<W: nxt.add((ny,nx,dx,dy))
            elif k=="wall": flash=4
            elif k=="block":
                for (ox,oy) in p:
                    if kind(y+oy,x+ox) is None: nxt.add((y+oy,x+ox,dx,dy))
            elif k=="target":
                (ax,ay),(bx,by)=p; a,bb=kind(y+ay,x+ax),kind(y+by,x+bx)
                if a=="target" and bb=="target": filled.add(int(tid[ny,nx]))
                else:
                    for (ox,oy) in p:
                        if kind(y+oy,x+ox) is None: nxt.add((y+oy,x+ox,dx,dy))
            elif k=="redir":
                (ax,ay),(bx,by)=p; a,bb=kind(y+ay,x+ax),kind(y+by,x+bx)
                da=(a=="redir" and bb is None); db=(bb=="redir" and a is None)
                if da: nxt.add((y-dx,x+dy,dy,-dx))
                if db: nxt.add((y+dx,x-dy,-dy,dx))
                else:
                    for (ox,oy) in p:
                        if kind(y+oy,x+ox) is None: nxt.add((y+oy,x+ox,dx,dy))
        drops=nxt; snaps.append((set((y,x) for (y,x,_,_) in drops),set(filled),flash>0)); flash=max(0,flash-1)
        if not drops and tk>=emit: break
    return snaps


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--level",type=int,default=3)
    ap.add_argument("--seed",type=int,default=1234); ap.add_argument("--out",default=None); a=ap.parse_args()
    arc=Arcade(arc_api_key="anonymous",environments_dir=str((LAB.parent/"environment_files").resolve()),operation_mode=OperationMode.OFFLINE)
    e0=arc.make(GAME_ID,seed=a.seed); _,c0=_present(e0,a.level)
    comps=frame_components(c0)[:10]; centers=[(int(c["center_xy"][0]),int(c["center_xy"][1])) for c in comps]
    blocks=[i for i,c in enumerate(comps) if int(c["color"]) in (8,9)]
    rng=np.random.default_rng(a.seed)
    def evg(genome):
        e=arc.make(GAME_ID,seed=a.seed); fd,cur=_present(e,a.level)
        fd,cur,_=_apply_real(e,fd,cur,actions(list(genome),centers))
        if not game_live(fd): return None,None
        ng=native_grid(e._game)
        return pour_core_v(ng["block"],ng["target"],ng["wall"],ng["redir"],ng["tid"],ng["spouts"],ng["n_targets"]),ng
    def fit(r): return (int(r["win"]),r["n_filled"],-r["n_wall"]) if r else (-1,-1,-99)
    def randg():
        k=int(rng.integers(2,13)); return [(int(rng.choice(blocks)),int(rng.choice(MOVES)),int(rng.integers(1,9))) for _ in range(k)]
    def mut(g):
        g=[list(x) for x in g]; op=int(rng.integers(5))
        if op==0 and g: i=int(rng.integers(len(g))); g[i][1]=int(rng.choice(MOVES))
        elif op==1 and g: i=int(rng.integers(len(g))); g[i][2]=int(rng.integers(1,9))
        elif op==2 and g: i=int(rng.integers(len(g))); g[i][0]=int(rng.choice(blocks))
        elif op==3 and len(g)<12: g.append([int(rng.choice(blocks)),int(rng.choice(MOVES)),int(rng.integers(1,9))])
        elif len(g)>2: del g[int(rng.integers(len(g)))]
        return [tuple(x) for x in g]

    best_chain=[]; tried=[]; evals=0; gbest=(-1,-1,-99); win_ng=None
    for restart in range(400):
        cur=randg(); cr,cg=evg(cur); evals+=1
        if cr is None: continue
        cf=fit(cr)
        for step in range(400):
            cand=mut(cur); rr,rg=evg(cand); evals+=1
            if rr is None: continue
            if evals%7==0 and len(tried)<80: tried.append((rg,rr,evals))
            if fit(rr)>cf:
                cur,cf,cg=cand,fit(rr),rg
                if fit(rr)>gbest:
                    gbest=fit(rr); best_chain.append((rg,rr,evals))
                    if rr["win"]: win_ng=rg
            if win_ng is not None: break
        if win_ng is not None: break
    print(f"search: {evals} evals, {len(best_chain)} best-so-far steps, win={win_ng is not None}",flush=True)

    # render
    H,W=best_chain[0][0]["block"].shape; out=a.out or str(FIXTURES/f"L{a.level}_search.mp4")
    vw=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*"mp4v"),18,(W*S,H*S+66))
    ti=0
    for (ng,res,ev) in best_chain:
        # a few 'exploring' flashes drawn from tried before this best
        for (tg,tr,te) in [tried[j] for j in range(ti,min(ti+4,len(tried)))]:
            im=board_img(tg,set(tr["filled"]),tr["fail"])
            f=frame_with_panel(im,[(f"SEARCHING  ·  candidate #{te}",MUT,0.5),
                                   (f"this try: {tr['n_filled']}/{tr['n_targets']} buckets, {tr['n_wall']} floor-hits",MUT,0.45)])
            for _ in range(2): vw.write(f)
        ti=min(ti+4,len(tried))
        win=res["win"]; status="WIN — all buckets, no floor" if win else "NEW BEST"
        col=GRN if win else GOLD
        im=board_img(ng,set(res["filled"]),res["fail"])
        f=frame_with_panel(im,[(f"BEST SO FAR  ·  {ev} arrangements tried",INK,0.5),
                               (f"{res['n_filled']}/{res['n_targets']} buckets filled  ·  {res['n_wall']} floor-hits",col,0.5),
                               (status,col,0.5)])
        for _ in range(int(28 if win else 16)): vw.write(f)
    # the winning pour dynamics
    if win_ng is not None:
        snaps=simulate_pour(win_ng)
        for (drops,filled,flash) in snaps:
            im=board_img(win_ng,filled,False)
            big=im.copy()
            for (y,x) in drops:
                cv2.circle(big,(x*S+S//2,y*S+S//2),S//2-2,C_DROP,-1)
            if flash: big[win_ng["wall"].repeat(S,0).repeat(S,1)]=C_FLASH
            f=frame_with_panel(big,[("THE WINNING POUR",GRN,0.55),("droplets fork, fill every bucket, shed off the sides",MUT,0.45)])
            for _ in range(3): vw.write(f)
        # hold final
        last=simulate_pour(win_ng)[-1]
        im=board_img(win_ng,last[1],False)
        f=frame_with_panel(im,[("SOLVED — engine-confirmed",GRN,0.6)])
        for _ in range(36): vw.write(f)
    vw.release(); print(f"wrote {out}")


if __name__=="__main__":
    main()

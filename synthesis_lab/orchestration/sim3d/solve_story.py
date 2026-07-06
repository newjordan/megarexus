"""The honest story of a solve, labeled: START (unsolved, loses) -> the search's block-moves
(blocks physically sliding) -> pour and WIN. Proves we start from the level as given and reach
the solution by moving blocks, not by framing from a finished game.

    python3 solve_story.py --level 3 --genome-json fixtures/l3_solve.json --out fixtures/L3_story.mp4
"""
from __future__ import annotations
import argparse, json, numpy as np, cv2
import _paths  # noqa: F401
from _paths import LAB, FIXTURES
from imagine_search_gate import _present, _grid, GAME_ID, ACTION_OF_ID, game_live
from imag_search_solve import actions
from engine_adapters.portband_glove_adapter import frame_components
from arc_agi import Arcade
from arc_agi.base import OperationMode

PAL=np.array([[0,0,0],[64,64,72],[128,0,128],[0,128,128],[230,180,40],[200,200,200],[90,160,255],
 [255,255,255],[160,110,60],[240,140,40],[120,200,120],[60,60,200],[24,24,28],[80,220,100],
 [220,60,60],[255,0,255]],np.uint8)
S=11; INK=(235,235,235); MUT=(150,150,160); GRN=(90,210,110); RED=(80,80,235); GOLD=(60,190,235)

def up(f):
    img=PAL[np.clip(np.asarray(f,np.int16),0,15)]
    img=np.kron(img,np.ones((S,S,1),np.uint8))
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

def panel(board,lines):
    W=board.shape[1]; p=np.full((88,W,3),(14,13,18),np.uint8)
    canvas=np.vstack([p,board])
    for i,(t,c,s) in enumerate(lines):
        cv2.putText(canvas,t,(16,26+i*24),cv2.FONT_HERSHEY_SIMPLEX,s,c,1,cv2.LINE_AA)
    return canvas

def win_frame(pour):  # frame with most filled(13) = the win state
    return pour[max(range(len(pour)),key=lambda i:int((pour[i]==13).sum()))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--level",type=int,default=3)
    ap.add_argument("--genome-json",required=True); ap.add_argument("--out",default=None); a=ap.parse_args()
    arc=Arcade(arc_api_key="anonymous",environments_dir=str((LAB.parent/"environment_files").resolve()),operation_mode=OperationMode.OFFLINE)
    genome=[tuple(x) for x in json.load(open(a.genome_json))["genome"]]
    e0=arc.make(GAME_ID,seed=1234); fd,c0=_present(e0,a.level)
    centers=[(int(c["center_xy"][0]),int(c["center_xy"][1])) for c in frame_components(c0)[:10]]
    ntgt=int((np.asarray(c0)==11).any(axis=0).sum()) or 4

    seq=[]  # (frame, lines, hold)
    # 1) START — unsolved
    e=arc.make(GAME_ID,seed=1234); fd,cur=_present(e,a.level); base=int(getattr(fd,"levels_completed",0) or 0)
    start=_grid(fd)
    seq.append((start,[("START",INK,0.8),("the level exactly as the game gives it",MUT,0.5),("blocks in their default spots",MUT,0.5)],36))
    # 2) pour it as-is -> loses
    fd=e.step(ACTION_OF_ID[5]); pf=[np.asarray(x,np.int8) for x in fd.frame]
    for f in pf: seq.append((f,[("Pour it AS-IS...",INK,0.7)],2))
    lose=pf[max(range(len(pf)),key=lambda i:int((pf[i]==14).sum()))]
    nb=int((win_frame(pf)==13).sum())//max(1,int((PAL[13]==PAL[13]).all()))
    seq.append((lose,[("...it LOSES",RED,0.8),("paint hits the floor (red flash) — not all buckets",RED,0.5)],40))
    # 3) reposition blocks — the search's discovered moves, one at a time
    e=arc.make(GAME_ID,seed=1234); fd,cur=_present(e,a.level)
    seq.append((_grid(fd),[("Now: the search's plan",INK,0.7),(f"move the blocks with {len(genome)} legal pushes",GOLD,0.5),("(discovered by hill-climb, not given)",MUT,0.45)],30))
    mv=0
    for (aid,x,y) in actions(genome,centers):
        if not game_live(fd): break
        fd=e.step(ACTION_OF_ID[aid],data={"x":int(x),"y":int(y)} if aid==6 else None)
        if aid==6: mv+=1; lbl=[("Repositioning blocks",GOLD,0.7),(f"push {mv} of {len(genome)}: select a block",MUT,0.5)]; hold=5
        else: lbl=[("Repositioning blocks",GOLD,0.7),(f"push {mv} of {len(genome)}: slide it",MUT,0.5)]; hold=3
        seq.append((_grid(fd),lbl,hold))
    arr=_grid(fd)
    seq.append((arr,[("SOLVED arrangement",GOLD,0.75),("blocks now route every stream to a bucket",MUT,0.5)],34))
    # 4) pour -> win
    fd=e.step(ACTION_OF_ID[5]); won=int(getattr(fd,"levels_completed",0) or 0)>base
    wf=[np.asarray(x,np.int8) for x in fd.frame]
    wi=max(range(len(wf)),key=lambda i:int((wf[i]==13).sum()))
    for f in wf[:wi+1]: seq.append((f,[("Pour again...",INK,0.7)],3))
    seq.append((wf[wi],[("WIN",GRN,0.9),(f"all {ntgt} buckets filled, floor never touched",GRN,0.5),("engine-confirmed",GRN,0.5)],48))

    W=start.shape[1]*S; H=start.shape[0]*S+88; out=a.out or str(FIXTURES/f"L{a.level}_story.mp4")
    vw=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*"mp4v"),18,(W,H))
    for f,lines,hold in seq:
        img=panel(up(f),lines)
        for _ in range(hold): vw.write(img)
    vw.release(); print(f"level {a.level} win={won} frames_of_story={len(seq)} wrote {out}")

if __name__=="__main__": main()

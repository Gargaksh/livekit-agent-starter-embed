#!/usr/bin/env python3
"""AhaTrove AT-001 licensed neural-voice synthesis. Manual, no-publish, zero-spend gate.

Use ONLY on confirmed no-incremental-charge compute. --dry-run never imports model.
All produced audio is REVIEW_ONLY: no release until independent word-level and
perceptual listening pass, rights clearance, and V4 final-film QA.
"""
import argparse, hashlib, json, os
from pathlib import Path

PHRASES=[
"How can a picture vanish with one shake?",
"Without using any ink at all?",
"This drawing toy has two little knobs.",
"Its secret is hiding behind the screen.",
"A thin layer of aluminum powder coats it.",
"Tiny plastic beads help the powder flow.",
"Now turn the left knob.",
"It slides a hidden tip sideways.",
"The right knob moves it vertically.",
"Turn both, and the tip traces curves.",
"But here’s the surprising part.",
"The tip doesn’t add a mark.",
"It pushes powder aside as it moves.",
"The darker inside becomes the visible line.",
"So how do you erase it?",
"You don’t need an eraser.",
"Flip the toy over, then shake gently.",
"The powder spreads back across the screen.",
"Every line disappears under that silver layer.",
"This toy draws by taking something away.",
]
def sha(b): return hashlib.sha256(b).hexdigest()
def ts(s):
    m=round(s*1000)
    return f"{m//3600000:02}:{m//60000%60:02}:{m//1000%60:02},{m%1000:03}"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--synthesize",action="store_true")
    p.add_argument("--out",default="AT001_VOICE_REVIEW_ONLY")
    p.add_argument("--pause",type=float,default=.12)
    p.add_argument("--seed",type=int,default=231123)
    args=p.parse_args()
    if not (0<=args.pause<=.35):raise ValueError("pause outside accepted range")
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    source=" ".join(PHRASES)
    locked={"source_sha256":sha(source.encode("utf-8")),"accepted_word_count":len(source.split()),
            "phrase_count":len(PHRASES),"model_license":"MIT per upstream; recheck weights revision",
            "local_compute_no_incremental_cost_confirmed":False,"commercial_voice_clearance":False,
            "authorized_to_publish":False}
    (dest/"input_lock.json").write_text(json.dumps(locked,indent=2),encoding="utf-8")
    if not args.synthesize:
        print(json.dumps({"status":"DRY_RUN","words":len(source.split()),
            "phrases":len(PHRASES),"narration_generated":False},indent=2));return
    import numpy as np,torch,torchaudio
    from chatterbox.tts import ChatterboxTTS
    torch.manual_seed(args.seed);torch.set_num_threads(min(4,os.cpu_count() or 2))
    model=ChatterboxTTS.from_pretrained(device="cpu")
    sr=48000;pad=np.zeros(round(sr*args.pause),dtype="float32")
    chunks=[];cues=[];now=0.;outfiles=[]
    for i,line in enumerate(PHRASES,1):
        with torch.no_grad(): y=model.generate(line)
        if y.ndim==2:y=y.mean(dim=0,keepdim=True)
        y=torchaudio.functional.resample(y,model.sr,sr)
        mono=y.reshape(-1).cpu().numpy().astype("float32")
        if len(mono)<int(.35*sr) or not np.isfinite(mono).all() or max(abs(mono))>1.:
            raise ValueError(f"Voice segment {i} failed sample QA")
        dur=len(mono)/sr; filename=f"phrase_{i:02d}.wav"
        torchaudio.save(str(dest/filename),torch.from_numpy(mono[None,:]),sr)
        outfiles.append({"phrase_id":i,"file":filename})
        cues.append({"id":f"AT001-S{i:02d}","start":round(now,4),"end":round(now+dur,4),"text":line})
        chunks.extend((mono,pad));now+=dur+len(pad)/sr
    output=np.concatenate(chunks[:-1])
    wav=dest/"AT001_NEURAL_REVIEW_ONLY_48k.wav"
    torchaudio.save(str(wav),torch.from_numpy(output[None,:]),sr)
    (dest/"phrases.srt").write_text("\n".join(
        f"{i}\n{ts(c['start'])} --> {ts(c['end'])}\n{c['text']}\n"
        for i,c in enumerate(cues,1)),encoding="utf-8")
    report={"status":"VOICE_READY_FOR_QA_NOT_CLEARED","duration":len(output)/sr,
            "duration_target_pass":45<=len(output)/sr<=60,
            "sample_rate":sr,"voice_sha256":sha(wav.read_bytes()),"phrase_alignment":cues,
            "model_reference_voice":"none; model default voice","commercial_voice_clearance":False,
            "independent_transcription_word_check":False,"listened_all_20_phrases":False,
            "compared_blind_to_Maheshwari_V3":False,"music_mixed":False,
            "final_H264_AAC_MP4":False,"youtube_upload_enabled":False}
    (dest/"REQUIRES_REVIEW.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"status":report["status"],"duration_s":round(report["duration"],2),
                      "duration_target_pass":report["duration_target_pass"]},indent=2))

if __name__=="__main__":main()

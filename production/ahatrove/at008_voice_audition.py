"""AhaTrove AT-008 — noncommercial evaluation-only ORIGINAL English audio.
No celebrity, existing narrator, third-party voice sample or imitation is used.
Outputs MUST remain publication HOLD pending expressive / acoustic / rights QA.
"""
import os, json, hashlib, time, wave, math
from pathlib import Path
import numpy as np
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

torch.set_num_threads(2)
out=Path("at008_neural_audition")
out.mkdir(exist_ok=True)
phrases=[
"Your checkout knows the price.",
"This barcode usually doesn’t.",
"So what is the scanner actually reading?",
"Some scanners sweep a beam of light",
"across the stripes.",
"Others capture an image.",
"Either way, the dark bars and light spaces",
"create a pattern of different widths.",
"A decoder turns that pattern into a number.",
"On a typical supermarket product,",
"that number identifies the item.",
"A check digit helps detect reading mistakes.",
"The checkout computer then looks up the identifier",
"in its database,",
"where it finds the product and its current price.",
"Here’s the clever part:",
"the store can change a price tomorrow",
"without printing a new barcode.",
"So that familiar beep isn’t the sound",
"of a price hiding in black ink.",
"It’s a computer finding",
"the right product."
]
# Preserve all words of the approved original fact-checked script.
assert len(phrases)==22
torch.manual_seed(258008)
print("LOADING Chatterbox model",flush=True)
t0=time.time()
model=ChatterboxTTS.from_pretrained(device="cpu")
sr=model.sr
assert sr in (24000, 48000),sr
print("MODEL LOADED",round(time.time()-t0,1),sr,flush=True)
chunks=[]; timings=[]; cursor=0
for idx,text in enumerate(phrases,1):
    torch.manual_seed(258008 + idx)
    t=time.time()
    with torch.inference_mode():
        wav=model.generate(text,exaggeration=0.60,cfg_weight=0.35)
    a=wav.detach().cpu().flatten().numpy().astype(np.float32)
    assert len(a)>sr//3 and np.isfinite(a).all()
    assert float(np.max(np.abs(a)))<1.01
    peak=float(np.max(np.abs(a)))
    rms=float(np.sqrt(np.mean(a*a)))
    # Natural sentence punctuation gets a slightly longer breath.
    gap=0.13 if text.endswith((".", "?", "!")) else 0.048
    if idx==len(phrases): gap=0.35
    fade=min(len(a)//12,int(sr*.012))
    if fade>0:
        a[:fade]*=np.linspace(0,1,fade)
        a[-fade:]*=np.linspace(1,0,fade)
    path=out/f"voice_{idx:02d}.wav"
    torchaudio.save(str(path),torch.tensor(a).unsqueeze(0),sr)
    start=cursor/sr
    end=start+len(a)/sr
    cursor+=len(a)+int(gap*sr)
    timings.append(dict(scene=idx,text=text,start_seconds=round(start,4),end_seconds=round(end,4),pause_seconds=gap,peak=round(peak,4),rms=round(rms,4),sample_rate=sr))
    chunks.extend([a,np.zeros(int(gap*sr),np.float32)])
    print("PHRASE",idx,round(end-start,2),"elapsed",round(time.time()-t,1),text,flush=True)
audio=np.concatenate(chunks)
torchaudio.save(str(out/"AT008_English_full_narration_REVIEW_ONLY.wav"),torch.tensor(audio).unsqueeze(0),sr)
(out/"AT008_phrase_boundaries.json").write_text(json.dumps(dict(schema=1,approval="REVIEW_ONLY_NOT_FOR_PUBLICATION",license_model="MIT (model/code claims need independent legal check)",reference_voice="none",voice_prompts="default pretrained voice only",speaker_cloning="no",source="GS1 institutional script, preproduction AT-008",duration_seconds=round(len(audio)/sr,4),sampling_rate=sr,scenes=timings),indent=2))
(out/"AT008_original_script.txt").write_text(" ".join(phrases)+"\n")
(out/"AT008_SOURCE_AND_REVIEW_NOTICE.txt").write_text("""AhaTrove AT-008 EVALUATION ONLY, NOT CLEARED FOR COMMERCIAL USE.
Upstream ResembleAI Chatterbox TTS model code and model card: MIT license:
https://github.com/resemble-ai/chatterbox
https://huggingface.co/ResembleAI/chatterbox
Voice: built-in pretrained default voice, NOT cloned from Maheshwari V3 or any individual.
No external music, outside footage, or third-party speech prompt.
Narration is UNAPPROVED until independent listening for expressive quality,
pronunciation, natural phrasing, technical voice continuity, match to approved script
and review of model/voice commercial provenance. No video/publishing authorization.
""")
for path in sorted(out.iterdir()):
    if path.is_file():
        print("HASH",hashlib.sha256(path.read_bytes()).hexdigest(),path.name)
print("TOTAL_SECONDS",round(len(audio)/sr,3),"GENERATION_TIME_SEC",round(time.time()-t0,1))

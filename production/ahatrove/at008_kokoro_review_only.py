"""AT-008 natural voice alternatives. REVIEW ONLY, no upload or release."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

SCRIPT = """Your checkout knows the price. This barcode usually doesn’t. So what is the scanner actually reading?

Some scanners sweep a beam of light across the stripes. Others capture an image. Either way, the dark bars and light spaces create a pattern of different widths. A decoder turns that pattern into a number.

On a typical supermarket product, that number identifies the item. A check digit helps detect reading mistakes. The checkout computer then looks up the identifier in its database, where it finds the product and its current price.

Here’s the clever part: the store can change a price tomorrow without printing a new barcode.

So that familiar beep isn’t the sound of a price hiding in black ink. It’s a computer finding the right product."""

OUT = Path("at008_kokoro_review_only")
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
SAMPLE_RATE = 24000

def generate_voice(code, name, speed):
    pipe = KPipeline(lang_code=code)
    segments, details = [], []
    cursor = 0
    for i,(graphemes,phonemes,audio) in enumerate(pipe(SCRIPT, voice=name, speed=speed, split_pattern=r"\n+")):
        arr=np.asarray(audio,dtype=np.float32).reshape(-1)
        if arr.size < SAMPLE_RATE//4:
            raise RuntimeError("TTS produced unexpectedly short audio")
        if np.max(np.abs(arr)) > 1.0: arr=arr/max(1.01,float(np.max(np.abs(arr))))
        segments.append(arr)
        details.append({"index":i,"source_text":graphemes,"sample_start":cursor,"sample_end":cursor+len(arr),"duration":round(len(arr)/SAMPLE_RATE,3)})
        cursor+=len(arr)
        if i < 10:
            spacer=np.zeros(int(SAMPLE_RATE*0.14),dtype=np.float32)
            segments.append(spacer)
            cursor+=len(spacer)
    # No fake speech placeholders; reject output below 25s or above 100s
    assembled=np.concatenate(segments)
    duration=float(len(assembled)/SAMPLE_RATE)
    if not(25.0 < duration < 100.0): raise RuntimeError(f"Voice duration outside bounds: {duration}")
    dest=OUT/f"AT008_{name}_FULL_DRAFT_REVIEW_ONLY.wav"
    sf.write(dest,assembled,SAMPLE_RATE,subtype="PCM_16")
    preview=OUT/f"AT008_{name}_FIRST16S_REVIEW_ONLY.wav"
    sf.write(preview,assembled[:16*SAMPLE_RATE],SAMPLE_RATE,subtype="PCM_16")
    return {"voice":name,"lang_code":code,"speed":speed,"duration_seconds":round(duration,3),"peak":float(np.max(np.abs(assembled))),"file":dest.name,"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"segments":details}

meta={"release_status":"REVIEW ONLY. HUMAN VOICE QUALITY AND SPEAKER/VOICE RIGHTS NOT SIGNED OFF", "source_script":SCRIPT, "script_sha256":hashlib.sha256(SCRIPT.encode()).hexdigest(),"model":"hexgrad/Kokoro-82M","model_license":"Apache-2.0","model_card":"https://huggingface.co/hexgrad/Kokoro-82M","voice_provenance":"https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md","model_card_cc_attribution":"Koniwa tnc CC-BY-3.0 and SIWIS CC-BY-4.0 are listed as training sources; preserve notices where applicable","results":[]}
for code,name,speed in [('a','af_heart',0.89),('b','bf_emma',0.92)]:
    meta["results"].append(generate_voice(code,name,speed))
(OUT/"VOICE_AUDITION_RIGHTS_AND_TIMINGS.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False))
print(json.dumps([{"voice":m["voice"],"duration":m["duration_seconds"],"sha":m["sha256"]} for m in meta["results"]], indent=2))

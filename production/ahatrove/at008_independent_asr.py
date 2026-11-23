"""Independent AhaTrove AT-008 transcript/time validation, review-only."""
import json,re,hashlib
from pathlib import Path
from difflib import SequenceMatcher
from faster_whisper import WhisperModel

SRC=Path("at008_kokoro_review_only")
META=json.loads((SRC/"VOICE_AUDITION_RIGHTS_AND_TIMINGS.json").read_text())
def normalize(s):
 s=s.lower().replace('’',"'").replace('‘',"'")
 return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?",s)
expected=normalize(META["source_script"])
model=WhisperModel("base.en",device="cpu",compute_type="int8",cpu_threads=2,num_workers=1)
out={"model":"Systran/faster-whisper, independent base.en ASR; timestamps estimates NOT forced verified ground truth","expected_script_sha256":META["script_sha256"],"expected_words":len(expected),"voices":[]}
for voice in META["results"]:
 path=SRC/voice["file"]
 segments,info=model.transcribe(str(path),language="en",beam_size=5,word_timestamps=True,vad_filter=False)
 words=[]
 for seg in segments:
  if seg.words:
   for w in seg.words:
    if w.word.strip():
     words.append({"start":round(float(w.start),3),"end":round(float(w.end),3),"text":w.word.strip(),"probability":round(float(w.probability),3)})
 transcript=" ".join(x["text"] for x in words)
 actual=normalize(transcript)
 matcher=SequenceMatcher(None,expected,actual)
 changes=[]
 for tag,i1,i2,j1,j2 in matcher.get_opcodes():
  if tag!='equal': changes.append({"tag":tag,"expected":" ".join(expected[i1:i2]),"heard_as":" ".join(actual[j1:j2])})
 result={"voice":voice["voice"],"source_audio_file":voice["file"],"matched_words_ratio":round(matcher.ratio(),4),"transcript":transcript,"asr_mismatch_flags":changes,"word_timestamps":words}
 out["voices"].append(result)
 print(voice["voice"],"ASR approx",len(actual),"/",len(expected),"ratio",matcher.ratio(),"mismatch",changes[:12],flush=True)
(SRC/"INDEPENDENT_ASR_WORD_TIMINGS.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))

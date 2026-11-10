"""Advanced (the owner's request of 2026-11-07): Krea 2 Turbo edits a photo from a plain
instruction, with the owner's own Krea 2 LoRA for the character.

Verified (2026-11-07):
- huggingface.co/conradlocke/krea2-identity-edit (revision f33b2629): Krea 2 in-context edit
  LoRAs, "Requires the ComfyUI-Krea2Edit node pack" (dual conditioning: the source as VAE
  tokens + the instruction encoded together with the source by Qwen3-VL). Rank-128 files
  "rank-reduced from the full files, near-identical quality". Recommended: Turbo, 8 steps,
  CFG 1, ref_boost 4 (General Edit V2) / 2 (Identity Edit V2), grounding_px 768, ≤ 2 MP.
  Licenses: V2 files free for personal / non-commercial use only (commercial use needs the
  author's license); v1.2 under the Krea 2 Community License Agreement. Both LoRAs at 1.0
  together give noise ("use about 0.6 each").
- github.com/lbouaraba/comfyui-krea2edit v2.0.0 (comfy/custom_nodes.py): Krea2EditModelPatch
  (model, source_latent; optional vae + source_image, target_latent, ref_boost, fit_mode,
  ref_position) and Krea2EditGroundedEncode (clip, prompt; optional image, grounding_px).
  Its example workflow: UNETLoader krea2_turbo_fp8_scaled, CLIPLoader qwen3vl_4b_fp8_scaled
  (type krea2), VAELoader qwen_image_vae, LoraLoaderModelOnly, EmptySD3LatentImage, KSampler
  euler / simple, cfg 1.
Measured on the studio's own runs: see docs/PROVIDERS.md (Advanced).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.comfy import enhance, krea2

FAMILY = "k2edit"
MODEL_KEY = "krea2-advanced"
NAME = "Krea 2 Advanced"
HF_REPO = "conradlocke/krea2-identity-edit"
HF_REVISION = "f33b26291c640e8ec50b0da420bfee3c9a660daa"


@dataclass(frozen=True)
class EditLora:
    key: str
    filename: str
    size: int
    sha256: str
    license: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{self.filename}"


EDIT_LORAS = {
    lora.key: lora
    for lora in (
        EditLora(
            "general_v2",
            "krea2_general_edit_v2_r128.safetensors",
            914_188_760,
            "fbca6ab74b9bef3db8de1d4a49b93d4fe7a778f5d4ae6995e0769ea8de736f00",
            "Krea 2 Identity Edit V2 License (non-commercial) + Krea 2 Community License",
        ),
        EditLora(
            "identity_v2",
            "krea2_identity_edit_v2_r128.safetensors",
            914_188_768,
            "2082515f1eb66532f5efb5343eb15048a29e3abe9f356e35e784cc60d00b45f7",
            "Krea 2 Identity Edit V2 License (non-commercial) + Krea 2 Community License",
        ),
        EditLora(
            "identity_v1_2",
            "krea2_identity_edit_v1_2_r128.safetensors",
            914_159_744,
            "f53db0bb4b081d638f196865cbc9f055379704fafb788336784fc1ccde18d825",
            "Krea 2 Community License",
        ),
    )
}
# The edit LoRA Advanced runs with, and its settings (the studio's own tests, PROVIDERS.md):
# ref_boost 4 keeps the photo best; with a character LoRA 2 lets the face become the
# character's (likeness up, the pose, clothes and background still kept).
DEFAULT_EDIT = "general_v2"


# A region edit (body shape: "breasts", "butt", "waist and hips"): SAM 3.1 (built-in
# nodes, the same file as Enhance photo) finds the part; ref_boost 0.05 only there. Measured:
# much stronger than ref_boost 0.6 on the whole photo, the rest stays (PROVIDERS.md).
SAM3 = enhance.SAM3
REGION_BOOST = 0.05
REGION_GROW = 40
REGION_THRESHOLD = 0.4
# Body parts the user can pick (Body), as SAM words; one word alone missed parts often in the
# studio's tests ("breasts" found nothing on a photo where "chest" did).
BODY_PARTS: dict[str, tuple[str, ...]] = {
    "bust": ("breasts", "chest"),
    # SAM 3.1's "buttocks" marked the top of the body (the chest, a T-shirt, a selfie arm) on
    # 10 of 10 of the owner's photos and "hips" a hand or the mouth; "butt" found the butt on
    # 8 of 10 and "bottom" the clothes over it on 9 (2026-11-08).
    "butt": ("butt", "bottom"),
    "waist": ("waist", "hips"),
    "belly": ("belly", "waist"),
    "legs": ("legs", "thighs"),
    "arms": ("arms", "shoulders"),
    # The whole figure (slimmer, curvier, like the character): let go more gently (0.2), at
    # 0.05 it restaged a selfie.
    "figure": ("breasts", "butt", "waist", "hips", "thighs"),
}
REGION_BOOSTS = {"figure": 0.2}
# Body: the character LoRA at half its strength in the edit (the face step still brings her
# face at full strength): likeness the same or higher on 4 photos (0.65/0.62/0.68/0.71 vs
# 0.64/0.58/0.67/0.71) and a slimmer waist no longer turned a crop top into a sports bra.
BODY_LORA = 0.5
# Clothes (Clothes): SAM finds them; edited with everything else held (hold_rest).
BODY_PARTS["clothes"] = ("clothes", "swimsuit")
HOLD_BOOST = 4.0
# Body edits: without it, a smaller bust came out without its bikini top (seen on the owner's
# photos); with it the clothes stay and follow the new shape.
KEEP_CLOTHES = "Her clothes stay on and look exactly the same, only fitting the new shape."
# Clothes and body edits, and character edits of the whole photo (Normal; a smaller bust
# restaged the owner's car selfie and lost her hand without them): a one-piece swimsuit, or
# the character alone, stood up a woman lying on a lounger in a tilted selfie; a hand on the
# cheek and a tongue out were lost. With these words (and the body held) she stayed lying
# and kept her hand and expression (2 of 2 seeds, 6 photos, 2026-11-07).
KEEP_POSE = "Keep her exact pose, body position and camera angle."
# Words asking for a new pose (English, and Slovak stems): then KEEP_POSE is left out.
# ponytail: a word list; a missed one only keeps the pose ("Strong" never adds the words).
POSE_WORDS = re.compile(
    r"\b(pose|sit\w*|seated|stand\w*|lie|lies|lying|lay\w*|kneel\w*|squat\w*|lean\w*|bend\w*"
    r"|turn(?:s|ed|ing)?\b(?!(?:\s+\w+){0,3}\s+into\b)|walk\w*|jump\w*|danc\w*|hug\w*|cross\w*|rais\w*|lift\w*|hands?|arms?|legs?"
    r"|póz\w*|sed\w*|stoj\w*|lež\w*|ľah\w*|kľač\w*|otoč\w*|ruk\w*|nohy|nohu|nohami|drep\w*|sadn\w*|posad\w*)\b",
    re.IGNORECASE,
)


# What is to stay ("Keep her pose and the background", "nechaj ju v tej polohe"): from such a
# word to the end of its sentence is not a request for a new pose. The studio's own rewrite of
# the owner's request ended "Keep her current facial expression, her pose, and the background
# exactly as they are." and the hold was dropped.
KEEP_WORDS = re.compile(
    r"\b(keep\w*|same|unchanged|stay\w*|remain\w*|don't|do not|without changing|as (it|they|she)"
    r" (is|are)|nechaj\w*|nemeň\w*|nemen\w*|zachov\w*|rovnak\w*|ost[aá]v\w*|ostan\w*)\b[^.!?;]*",
    re.IGNORECASE,
)


def asks_for_pose(text: str) -> bool:
    return bool(POSE_WORDS.search(KEEP_WORDS.sub(" ", text)))


# Text to image with the 18+ switch on and words asking for her naked: Krea 2 Turbo put a
# thong, a bra or a bikini on "completely nude" in 5 of 9 prompts (her LoRA on every step) and
# in 2 of 9 with this sentence added (2026-11-10, PROVIDERS.md › Advanced). "nude" as a colour
# (lipstick, heels) is not such a word.
NUDE_NOTE = "She is fully naked: bare breasts, bare hips and bare skin, nothing on her body."
NUDE_WORDS = re.compile(
    r"\b(naked|nah[áéýú]\w*|nahot\w*|bez obleč\w*|nude(?!\s+(?:lip\w*|heel\w*|colou?r\w*"
    r"|tone\w*|make-?up|nail\w*|shoe\w*|dress\w*|bra\w*|panties|lingerie|top\b|tops|bodysuit\w*"
    r"|swim\w*|underwear|beige)))\b",
    re.IGNORECASE,
)


def asks_for_nude(text: str) -> bool:
    return bool(NUDE_WORDS.search(text))


# 18+ text to image naming her intimate area: Turbo still covered it with a thong on all fours
# (1 of 8 with NUDE_NOTE); with this sentence 8 of 8 were bare, nothing else changed
# (2026-11-10).
VULVA_NOTE = "Her bare vulva is clearly visible, nothing covers it."
VULVA_WORDS = re.compile(
    r"\b(pussy|vagina\w*|vulva\w*|labia\w*|kund\w*|pičk\w*|piča|piču|vagín\w*)\b", re.IGNORECASE
)


def names_vulva(text: str) -> bool:
    return bool(VULVA_WORDS.search(text))


# Text to image, a selfie seen from the front: with the pose composed first the phone was held
# over her face (a squat selfie); this sentence moved it beside her face (2026-11-10). Not for
# views from behind (she looks back, or her face is not in the photo).
FACE_NOTE = "Her face is fully visible, not hidden by the phone."
SELFIE_WORDS = re.compile(r"\b(selfie\w*|phone\w*|mirror\w*|mobil\w*|zrkadl\w*)\b", re.IGNORECASE)
BACK_WORDS = re.compile(
    r"\b(behind|back to|from the back|rear|facing away|shoulder|looking back|back view"
    r"|zozadu|zo zadu|chrbt\w*|otočen\w*)\b",
    re.IGNORECASE,
)


def front_selfie(text: str) -> bool:
    return bool(SELFIE_WORDS.search(text)) and not BACK_WORDS.search(text)


HAIR_ASK = re.compile(r"\b(hair\w*|vlas\w*)\b", re.IGNORECASE)


def asks_for_hair(text: str) -> bool:
    """Whether the words give her new hair ("… long straight black hair"; not "keep her hair")."""
    return bool(HAIR_ASK.search(KEEP_WORDS.sub(" ", text)))


# A body change asked for on the whole photo ("…, long black hair, a much larger bust and a
# rounder butt"): there her body is held (her pose), so the shape is changed in a second pass
# like Body, with the body words alone (body_change). On 8 of the owner's photos (2026-11-08)
# the whole instruction in that pass barely changed the shape; the body words alone made the
# bust and butt clearly bigger, the pose kept (one framing change, caught by k2edit.moved).
# ponytail: word lists (English, Slovak stems); a missed word only skips the second pass.
PART_NOUNS = {
    "bust": r"bust\w*|breast\w*|boobs?|chest|cleavage|prs\w*|poprs\w*|hrud\w*",
    "butt": r"butt|butts|buttock\w*|booty|glutes?|hips?|zadok|zadk\w*|zadoč\w*|boky|bokov|bokmi",
    "waist": r"waist\w*|pás|pása|pásom",
    "belly": r"belly|stomach|tummy|brucho|bruch\w*",
    "legs": r"legs?|thighs?|nohy|nohami|stehn\w*",
    "arms": r"arms?|paž\w*",
    "figure": r"figure|body|physique|hourglass|postav\w*|telo|tela|telom",
}
_PARTS = {part: re.compile(rf"\b({words})\b", re.IGNORECASE) for part, words in PART_NOUNS.items()}
SHAPE_WORDS = re.compile(
    r"\b(big\w*|larg\w*|small\w*|slim\w*|thin\w*|skinn\w*|full\w*|round\w*|wide\w*|wider|narrow\w*"
    r"|long\w*|short\w*|curv\w*|toned|muscular|fat\w*|weight|size|huge|tiny|petite|hourglass"
    r"|väčš\w*|menš\w*|štíhl\w*|plnš\w*|okrúhl\w*|dlhš\w*|kratš\w*|širš\w*|užš\w*|chudš\w*"
    r"|zväčš\w*|zmenš\w*)\b",
    re.IGNORECASE,
)
# Clauses about something else ("give her long black hair") stay out of the body pass.
OTHER_WORDS = re.compile(
    r"\b(hair\w*|face|facial|eyes?|lips|mouth|nose|smile|expression|background|light\w*|makeup"
    r"|vlas\w*|tvár\w*|oči|očí|pery|úsmev\w*|pozad\w*|svetl\w*)\b",
    re.IGNORECASE,
)
FIGURE_WORDS = re.compile(
    r"\b(slim\w*|thin\w*|skinn\w*|curv\w*|fat\w*|weight|muscular|toned|petite|štíhl\w*|chud\w*)\b",
    re.IGNORECASE,
)
LEADING_VERB = re.compile(
    r"(give|make|change|add|turn|let|enlarge|reduce|increase|decrease|shrink|slim|daj|sprav"
    r"|zmeň|zmen|zväčši|zmenši|urob|nech)\b",
    re.IGNORECASE,
)
MAX_REGION_WORDS = 6
# The body pass: her arms and hands out of the change, grown (a hand on the cheek went with
# its wrist in the region), her face held. Let go (0.05) or held only at 1 with the face free,
# the body pass stood up a lying woman, lost a hand on a cheek and restaged a woman leaning
# on a mirror; held like this the pose stayed on 11 of 12 photos, the bust clearly bigger,
# the butt a little (2026-11-08); the twelfth, a selfie arm SAM's "buttocks" took for the butt,
# became her bust until the butt's words were changed (BODY_PARTS).
# Her feet too: a foot under a kneeling woman's butt came out melted into it (2026-11-08).
# The body pass also says the body stays natural, and uses her LoRA at full strength (her own
# figure): the studio's rewrite of "bigger bust" as "much larger bust" made an already full
# bust grotesque (the owner's rollers photo); with these words and her LoRA it came out natural
# on 8 of 8 photos (2026-11-08).
BODY_NATURAL = "Her body stays natural and realistic, in proportion."
LIMB_WORDS = ("arm", "hand", "feet")
LIMB_GROW = 40
# Put back tight (paste_held): grown 40 px, the old bust's edge beside an arm came back as a
# pale halo around the new one (the owner's mirror selfie).
PASTE_GROW = 8


_PIECES = re.compile(r",\s*|\s+and\s+", re.IGNORECASE)


def _pieces(text: str) -> list[str]:
    """Clauses: split at commas and "and", without a hanging "and"/"but"/"or"."""
    cut = (
        re.sub(r"^(and|but|or)\s+|\s+(and|but|or)$", "", p.strip(), flags=re.I)
        for p in _PIECES.split(text.strip(" .!?;"))
    )
    return [p for p in cut if p]


def body_change(prompt: str) -> tuple[str, str, str] | None:
    """(the instruction's body words alone, the region: parts joined by "+", the instruction
    without them) when it changes the body's shape; None otherwise (a pose: "cross her legs",
    hands on her hips). The edit gets the rest: with the body words a lying woman in a tilted
    selfie stood up to show her new bust (2 of 2 runs), held body and all."""
    kept: list[str] = []
    parts: list[str] = []
    rest: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+", prompt.strip()):
        body = [
            c
            for c in _pieces(KEEP_WORDS.sub(" ", sentence))
            if any(rx.search(c) for rx in _PARTS.values())
            or (SHAPE_WORDS.search(c) and not OTHER_WORDS.search(c))
        ]
        named = [p for p, rx in _PARTS.items() if any(rx.search(c) for c in body)]
        if not named and body and FIGURE_WORDS.search(" ".join(body)):
            named = ["figure"]  # "make her slimmer"
        if not (named and SHAPE_WORDS.search(" ".join(body))):
            rest.append(sentence)
            continue
        kept.append(", ".join(body))
        parts += [p for p in named if p not in parts]
        left = [p for p in _pieces(sentence) if not any(p.startswith(b) for b in body)]
        if left:
            rest.append(f"{left[0][0].upper()}{left[0][1:]}{''.join(', ' + p for p in left[1:])}.")
    if not kept:
        return None
    if "figure" in parts and len(parts) > 1:
        # "a slimmer figure with a bigger bust": the named parts, and the waist for the figure.
        parts = [p for p in parts if p != "figure"] + ([] if "waist" in parts else ["waist"])
    words = {w for p in parts for w in BODY_PARTS[p]}
    region = "figure" if len(words) > MAX_REGION_WORDS else "+".join(parts)
    # "give her long hair, a much larger bust": the body clause without its verb gets one.
    kept = [k if LEADING_VERB.match(k) else f"give her {k[0].lower()}{k[1:]}" for k in kept]
    text = ". ".join(k[0].upper() + k[1:] for k in kept)
    return f"{text}.", region, " ".join(rest)


# Hair: SAM 3.1 finds it on the whole photo and the edit is let go there (like Body): a new
# colour or a short cut on a small person changed little at ref_boost 4 (measured).
HAIR_WORDS = ("hair",)
HANDS_NODE = "64"
SAM_NODES = (
    "SAM3_Detect",
    "GrowMask",
    "MaskComposite",
    "MaskToImage",
    "PreviewImage",
    "InvertMask",
    "SetLatentNoiseMask",  # the face step keeps an open mouth (MOUTH_WORDS)
    "ImageBlur",  # the detail pass (detail_graph)
    "ImageToMask",
    "ImageCompositeMasked",
)
# "strong": ref_boost 0.6 lets bigger changes through (body shape, a new pose) while the
# scene still holds; at 2-4 the body barely changes (measured, PROVIDERS.md).
STRENGTHS = ("normal", "strong")


def settings_for(character: bool, strength: str = "normal", face: bool = False) -> dict[str, Any]:
    """Face only (the face crop): 1 keeps the person, 0.6 for hard expressions (one eyebrow,
    gaze); the whole photo: 4 (2 with a character LoRA), 0.6 strong (measured)."""
    if face:
        boost = 0.6 if strength == "strong" else 1.0
    else:
        boost = 0.6 if strength == "strong" else 2.0 if character else 4.0
    return {"steps": 8, "ref_boost": boost, "grounding_px": 768}


# Face only: the crop's mouth may grow a hand (a kiss) that the oval paste would cut off.
FACE_ONLY_NOTE = "Change only the face. Do not add hands."
# Face only: an oval around the face itself goes back (a little wider than the face step's,
# for an open mouth or raised brows). The square's own centred ellipse missed half of a face
# at the photo's edge (the square is moved inside the photo): a doubled eye and a seam, and
# hair from the crop beside the face (the owner's car selfie, 2026-11-07).
FACE_ONLY_GROW = 0.25


NODES = ("Krea2EditModelPatch", "Krea2EditGroundedEncode")
REQUIRED_NODES = (
    *NODES,
    "EmptySD3LatentImage",
    "LoraLoaderModelOnly",
    "ImageScale",
    "KSamplerAdvanced",
)
TURBO = krea2.VARIANTS["turbo"].known_files[0]  # krea2_turbo_fp8_scaled.safetensors
REF_POSITIONS = ("aligned (trained)", "beside (right of target)")
MAX_PROMPT = 2000


def _files(keys: tuple[str, ...]) -> list[tuple[str, str, str, float, str | None]]:
    """(url, folder, filename, size in GB, sha256) of everything Advanced needs."""
    out = [
        (f.url, f.folder, f.filename, f.size_gb, None)
        for f in (krea2.TEXT_ENCODER, krea2.VAE, TURBO)
    ]
    # SAM 3.1 (Face only keeps the hands; Body finds the part): Enhance photo's file.
    out.append((SAM3.url, SAM3.folder, SAM3.filename, SAM3.size / 1e9, None))
    for key in keys:
        e = EDIT_LORAS[key]
        out.append((e.url, "loras", e.filename, e.size / 1e9, e.sha256))
    return out


def downloads(keys: tuple[str, ...] = (DEFAULT_EDIT,)) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for url, folder, filename, _, sha in _files(keys):
        item: dict[str, Any] = {"url": url, "folder": folder, "filename": filename}
        if sha:
            item["sha256"] = sha
        out.append(item)
    return out


def sizes_gb(keys: tuple[str, ...] = (DEFAULT_EDIT,)) -> dict[str, float]:
    return {filename: size for _, _, filename, size, _ in _files(keys)}


def catalog_json() -> dict[str, Any]:
    edit = EDIT_LORAS[DEFAULT_EDIT]
    return {
        "name": NAME,
        "model_key": MODEL_KEY,
        "files": [
            {"folder": folder, "filename": filename, "size": round(size * 1e9)}
            for _, folder, filename, size, _ in _files((DEFAULT_EDIT,))
        ],
        "total_bytes": round(sum(sizes_gb().values()) * 1e9),
        "edit_lora": edit.filename,
        "edit_license": edit.license,
        "repo": f"https://huggingface.co/{HF_REPO}",
        "max_prompt": MAX_PROMPT,
    }


def detect(object_info: dict[str, Any]) -> dict[str, Any]:
    """Which of Advanced's nodes and files this ComfyUI has."""

    def choices(node: str, field: str) -> list[str]:
        spec = (object_info.get(node) or {}).get("input", {}).get("required", {}).get(field)
        return list(spec[0]) if spec and isinstance(spec[0], list) else []

    unets = choices("UNETLoader", "unet_name")
    encoders = choices("CLIPLoader", "clip_name")
    vaes = choices("VAELoader", "vae_name")
    loras = choices("LoraLoaderModelOnly", "lora_name")
    sam = SAM3.filename in choices("CheckpointLoaderSimple", "ckpt_name")
    edit = {k: e.filename in loras for k, e in EDIT_LORAS.items()}
    missing_nodes = [n for n in REQUIRED_NODES if n not in object_info]
    missing_files = [
        f.filename
        for f, found in (
            (TURBO, unets),
            (krea2.TEXT_ENCODER, encoders),
            (krea2.VAE, vaes),
        )
        if f.filename not in found
    ]
    return {
        "missing_nodes": missing_nodes,
        "missing_files": missing_files,
        "edit_loras": edit,
        # Face only and Body need SAM 3.1 (and its nodes, built into ComfyUI 0.35).
        "sam": sam and all(n in object_info for n in SAM_NODES),
        "ready": not missing_nodes and not missing_files and edit[DEFAULT_EDIT],
    }


MOVE_SHIFT = 0.12
MOVE_SIZE = (0.7, 1.45)


def moved(
    before: tuple[float, float, float, float],
    after: tuple[float, float, float, float],
    size_before: tuple[int, int],
    size_after: tuple[int, int],
) -> bool:
    """Whether the person moved between the photo and the edit: her face's centre shifted by
    more than MOVE_SHIFT of the picture, or its size changed beyond MOVE_SIZE."""
    (x0, y0, w0, h0), (x1, y1, w1, h1) = before, after
    (W0, H0), (W1, H1) = size_before, size_after
    dx = (x1 + w1 / 2) / W1 - (x0 + w0 / 2) / W0
    dy = (y1 + h1 / 2) / H1 - (y0 + h0 / 2) / H0
    ratio = (max(w1, h1) / max(W1, H1)) / max(1e-6, max(w0, h0) / max(W0, H0))
    return abs(dx) > MOVE_SHIFT or abs(dy) > MOVE_SHIFT or not MOVE_SIZE[0] <= ratio <= MOVE_SIZE[1]


def place(x: float, y: float) -> str:
    """Where a point (shares of the image) is, in words."""
    if x < 0.34:
        return "left edge"
    if x > 0.66:
        return "right edge"
    return "top" if y < 0.4 else "bottom" if y > 0.6 else "middle"


def other_face_instruction(who: str, side: str, main_side: str, note: str = "") -> str:
    """A second pass for another face of her (a mirror reflection): the edit turns the main
    person into the character and leaves a reflection as it was; named by its place, with the
    user's own instruction repeated (without it the reflection's hair stayed), it changes too
    (measured on the owner's elevator photo, PROVIDERS.md)."""
    where = "in the middle" if main_side == "middle" else f"at the {main_side}"
    return (
        f"The person at the {side} of the photo is the same woman (her reflection in a mirror):"
        f" turn her into {who} too, like the woman {where}. {note.strip()} Keep her face, her"
        " hands and her pose visible as they are."
    ).replace("  ", " ")


def names(prompt: str, trigger: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(trigger)}(?!\w)", prompt, re.IGNORECASE) is not None


def _sam(wf: dict[str, Any], node: str, words: str, image: list[Any]) -> list[Any]:
    """SAM 3.1 (loaded at node "40") finding `words` in the image; returns the mask's link."""
    text, detect = node, str(int(node) + 1)
    wf[text] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["40", 1], "text": words}}
    wf[detect] = {
        "class_type": "SAM3_Detect",
        "inputs": {
            "model": ["40", 0],
            "image": image,
            "conditioning": [text, 0],
            "threshold": REGION_THRESHOLD,
            "refine_iterations": 2,
            "individual_masks": False,
        },
    }
    return [detect, 0]


HAIR_NODE = "137"  # where her old hair was (grown), shown for keep_unchanged
PART_NODE = "189"  # the part a region edit changes, shown for keep_unchanged
HAIR_BLUR_GROW = 32
HAIR_BLUR_PASSES = 1
HAIR_BLUR_BG_PASSES = 3
HAIR_BLUR_FEATHER = 15  # ImageBlur radius of the blur mask's edge
HAIR_BLUR_FACE_GAP = 32  # the hair this close to her face stays sharp


def _feather(wf: dict[str, Any], node: str, mask: list[Any], radius: int) -> list[Any]:
    """`mask` with soft edges (nodes node .. node+2); returns the soft mask's link."""
    a, b, c = node, str(int(node) + 1), str(int(node) + 2)
    wf[a] = {"class_type": "MaskToImage", "inputs": {"mask": mask}}
    wf[b] = {
        "class_type": "ImageBlur",
        "inputs": {"image": [a, 0], "blur_radius": radius, "sigma": radius / 2},
    }
    wf[c] = {"class_type": "ImageToMask", "inputs": {"image": [b, 0], "channel": "red"}}
    return [c, 0]


HAIR_GROW = 48


def _show_hair(wf: dict[str, Any], hair: list[Any]) -> None:
    """The photo's hair (grown) as a preview: there the edit's pixels are always taken
    (keep_unchanged): fine blond curls over a pale wall differed too little from the new wall
    and came back as a ghost beside her new black hair (the owner's mirror selfie)."""
    wf["135"] = {
        "class_type": "GrowMask",
        "inputs": {"mask": hair, "expand": HAIR_GROW, "tapered_corners": True},
    }
    wf["136"] = {"class_type": "MaskToImage", "inputs": {"mask": ["135", 0]}}
    wf[HAIR_NODE] = {"class_type": "PreviewImage", "inputs": {"images": ["136", 0]}}


def without_masks(workflow: dict[str, Any]) -> dict[str, Any]:
    """The edit graph without SAM 3.1 and everything that uses it (masks, previews, the blurred
    reference): the plain photo is the reference, one ref_boost everywhere. The reflection pass
    edits the whole result this way; a leftover preview of a removed mask made the graph
    invalid and the reflection stayed as it was."""
    graph = {k: {**v, "inputs": dict(v["inputs"])} for k, v in workflow.items()}
    graph["12"]["inputs"]["pixels"] = ["101", 0]
    graph["13"]["inputs"]["source_image"] = ["101", 0]
    graph["4"]["inputs"]["image"] = ["101", 0]
    graph["13"]["inputs"].pop("ref_boost_mask", None)
    gone = {"40"}
    changed = True
    while changed:
        changed = False
        for key, node in graph.items():
            links = [v for v in node["inputs"].values() if isinstance(v, list) and v]
            if key not in gone and any(str(v[0]) in gone for v in links):
                gone.add(key)
                changed = True
    return {k: v for k, v in graph.items() if k not in gone}


def with_character(prompt: str, trigger: str, class_word: str, hair: str = "") -> str:
    """The instruction with the LoRA's character asked for first, so the person in the photo
    becomes the character (a trailing "The woman is ..." changed nothing in the studio's tests
    when the instruction was about something else, e.g. the background). The edit keeps the
    photo's hair unless told: `hair` (from the AI model, e.g. "long straight black") is
    named when known."""
    if names(prompt, trigger):
        return prompt
    who = class_word or "person"
    look = f", with {hair.strip().rstrip('.')} hair" if hair.strip() else ""
    return f"Turn the {who} into {trigger} {who}{look}. {prompt.strip()}".strip()


def build_edit(
    *,
    image: str,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    edit_lora: str,
    edit_strength: float = 1.0,
    loras: list[tuple[str, float]] | None = None,
    steps: int = 8,
    ref_boost: float = 4.0,
    grounding_px: int = 768,
    image_b: str | None = None,
    ref_position: str = REF_POSITIONS[0],
    prefix: str = "ai-image-studio-advanced",
    region: str | None = None,
    keep_hands: bool = False,
    hold_body: bool = False,
    hold_rest: bool = False,
    keep_limbs: bool = False,
    blur_hair: bool = False,
) -> dict[str, Any]:
    """The API-format edit graph: the photo (scaled to width x height) is the reference the
    edit LoRA keeps, the instruction is encoded together with it, the owner's LoRAs follow the
    edit LoRA. An optional second photo (image_b) is the person for two-image edits (the first
    photo is then the scene, as the LoRAs were trained)."""
    if ref_position not in REF_POSITIONS:
        raise ValueError("Unknown reference position")
    wf: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": TURBO.filename, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": krea2.TEXT_ENCODER.filename,
                "type": krea2.FAMILY,
                "device": "default",
            },
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": krea2.VAE.filename}},
        "20": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": EDIT_LORAS[edit_lora].filename,
                "strength_model": edit_strength,
            },
        },
    }
    model_ref: list[Any] = ["20", 0]
    for index, (name, strength) in enumerate(loras or []):
        node_id = str(21 + index)
        wf[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_ref, "lora_name": name, "strength_model": strength},
        }
        model_ref = [node_id, 0]

    def scaled(node: str, name: str, w: int, h: int) -> list[Any]:
        wf[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf[node + "1"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": [node, 0],
                "upscale_method": "lanczos",
                "width": w,
                "height": h,
                "crop": "center",
            },
        }
        return [node + "1", 0]

    src = scaled("10", image, width, height)
    wf["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": src, "vae": ["3", 0]}}
    wf["6"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    patch: dict[str, Any] = {
        "model": model_ref,
        "source_latent": ["12", 0],
        "ref_boost": ref_boost,
        "ref_boost_a": 1.0,
        "fit_mode": "fit",
        "vae": ["3", 0],
        "source_image": src,
        "target_latent": ["6", 0],
        "ref_position": ref_position,
        "ref_position_a": REF_POSITIONS[0],
    }
    encode: dict[str, Any] = {
        "clip": ["2", 0],
        "prompt": prompt,
        "image": src,
        "grounding_px": grounding_px,
        "system_prompt": "",
    }
    if image_b:
        # Two images: the scene first (aligned), the person second; ref_boost acts on the last.
        person = scaled("30", image_b, width, height)
        wf["32"] = {"class_type": "VAEEncode", "inputs": {"pixels": person, "vae": ["3", 0]}}
        patch |= {"source_latent_b": ["32", 0], "source_image_b": person}
        encode["image_b"] = person
    if region or keep_hands or hold_body:
        wf["40"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3.filename}}
    if hold_body and not region:
        # The character's body (SAM "woman" minus her hair and face) held at HOLD_BOOST, the
        # rest (hair, face, background) at 1: her pose stays even when new long hair is asked
        # for; letting the hair go instead stood up a woman lying on a sun lounger.
        hair = _sam(wf, "120", "hair", src)
        wf["126"] = {
            "class_type": "GrowMask",
            "inputs": {"mask": hair, "expand": 16, "tapered_corners": True},
        }
        wf["127"] = {
            "class_type": "MaskComposite",
            "inputs": {
                "destination": _sam(wf, "110", "woman", src),
                "source": ["126", 0],
                "x": 0,
                "y": 0,
                "operation": "subtract",
            },
        }
        wf["128"] = {
            "class_type": "MaskComposite",
            "inputs": {
                "destination": ["127", 0],
                "source": _sam(wf, "123", "face", src),
                "x": 0,
                "y": 0,
                "operation": "subtract",
            },
        }
        # Her hands held too: a hand holding up her hair went with the old hair (2026-11-08).
        wf["134"] = {
            "class_type": "GrowMask",
            "inputs": {"mask": _sam(wf, "132", "hand", src), "expand": 16, "tapered_corners": True},
        }
        # Not the hair in her hand (a blond tuft she held up stayed blond in her new hair).
        wf["138"] = {
            "class_type": "MaskComposite",
            "inputs": {
                "destination": ["134", 0],
                "source": hair,
                "x": 0,
                "y": 0,
                "operation": "subtract",
            },
        }
        wf["131"] = {
            "class_type": "MaskComposite",
            "inputs": {
                "destination": ["128", 0],
                "source": ["138", 0],
                "x": 0,
                "y": 0,
                "operation": "add",
            },
        }
        patch |= {"ref_boost": HOLD_BOOST, "ref_boost_mask": ["131", 0]}
        _show_hair(wf, hair)
        if blur_hair:
            # New hair: the old hair (minus her hands and face) blurred in the reference, so
            # its fine strands are not copied: blond waves over a shoulder came out as thin
            # dark lines on the skin and blond ends stayed on the floor beside her new hair;
            # blurred, the skin was clean, a tank top strap came back and a hand holding up
            # her hair stayed (8 of the owner's photos, 2026-11-08). Grown 32 px, blurred once:
            # blurred three times, the old brown hair over a forearm came out as dark smudges
            # on the skin; grown 48 px, a choker beside the hair went.
            wf["180"] = {
                "class_type": "GrowMask",
                "inputs": {"mask": hair, "expand": HAIR_BLUR_GROW, "tapered_corners": True},
            }
            wf["181"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": ["180", 0],
                    "source": ["138", 0],
                    "x": 0,
                    "y": 0,
                    "operation": "subtract",
                },
            }
            # Not near her face either: blurred up to her cheeks, a smile with teeth closed
            # (2 of 2 seeds; sharp, it stayed).
            wf["199"] = {
                "class_type": "GrowMask",
                "inputs": {
                    "mask": ["124", 0],
                    "expand": HAIR_BLUR_FACE_GAP,
                    "tapered_corners": True,
                },
            }
            wf["182"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": ["181", 0],
                    "source": ["199", 0],
                    "x": 0,
                    "y": 0,
                    "operation": "subtract",
                },
            }
            # Not her accessories (a sleep mask on the hair became a pink headwrap when
            # blurred; SAM's "accessory" found it, a choker and a necklace, not hair rollers).
            # ... nor her jewelry ("accessory" missed a charm bracelet by her hair, which then
            # came out melted).
            wf["198"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": _sam(wf, "190", "accessory", src),
                    "source": _sam(wf, "196", "jewelry", src),
                    "x": 0,
                    "y": 0,
                    "operation": "add",
                },
            }
            wf["192"] = {
                "class_type": "GrowMask",
                "inputs": {"mask": ["198", 0], "expand": 8, "tapered_corners": True},
            }
            wf["193"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": ["182", 0],
                    "source": ["192", 0],
                    "x": 0,
                    "y": 0,
                    "operation": "subtract",
                },
            }
            # Over her (skin, clothes) once; over the background HAIR_BLUR_BG_PASSES times: a
            # light blob of old hair over a wall, blurred once, came back as a see-through
            # outline of the old hair.
            soft = [src]
            for node in ("183", "185", "186")[: max(HAIR_BLUR_PASSES, HAIR_BLUR_BG_PASSES)]:
                wf[node] = {
                    "class_type": "ImageBlur",
                    "inputs": {"image": soft[-1], "blur_radius": 31, "sigma": 10.0},
                }
                soft.append([node, 0])
            # Soft edges: a hard edge between the blurred old hair and the sharp wall came back
            # as a thin see-through outline (and a smudge on an elbow).
            for node, op, soft_node in (("194", "multiply", "210"), ("195", "subtract", "214")):
                wf[node] = {
                    "class_type": "MaskComposite",
                    "inputs": {
                        "destination": ["193", 0],
                        "source": ["111", 0],
                        "x": 0,
                        "y": 0,
                        "operation": op,
                    },
                }
                _feather(wf, soft_node, [node, 0], HAIR_BLUR_FEATHER)
            wf["184"] = {
                "class_type": "ImageCompositeMasked",
                "inputs": {
                    "destination": src,
                    "source": soft[HAIR_BLUR_PASSES],
                    "x": 0,
                    "y": 0,
                    "resize_source": False,
                    "mask": ["212", 0],
                },
            }
            wf["187"] = {
                "class_type": "ImageCompositeMasked",
                "inputs": {
                    "destination": ["184", 0],
                    "source": soft[HAIR_BLUR_BG_PASSES],
                    "x": 0,
                    "y": 0,
                    "resize_source": False,
                    "mask": ["216", 0],
                },
            }
            blurred = ["187", 0]
            wf["12"]["inputs"]["pixels"] = blurred
            patch["source_image"] = blurred
            encode["image"] = blurred
    if region:
        # Only that part of the photo (SAM 3.1 finds it from the words, grown a little so a
        # bigger shape has room) is let go: the edit hardly looks at it there, and keeps the
        # rest as at ref_boost 1. Two words per part (BODY_PARTS), added together.
        # Several parts at once ("bust+butt", k2edit.body_change): their words together.
        words = (
            HAIR_WORDS
            if region == "hair"
            else tuple(dict.fromkeys(w for p in region.split("+") for w in BODY_PARTS.get(p, (p,))))
        )
        masks = [_sam(wf, str(50 + 3 * i), word, src) for i, word in enumerate(words)]
        mask = masks[0]
        for i, other in enumerate(masks[1:]):
            node = str(70 + i)
            wf[node] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": mask,
                    "source": other,
                    "x": 0,
                    "y": 0,
                    "operation": "add",
                },
            }
            mask = [node, 0]
        wf["43"] = {
            "class_type": "GrowMask",
            "inputs": {"mask": mask, "expand": REGION_GROW, "tapered_corners": True},
        }
        if region == "hair":
            _show_hair(wf, mask)
        # The face is never in the let-go part (a grown hair mask reached it and the face
        # changed with a new hair colour): it stays held as at ref_boost 1.
        face = _sam(wf, "80", "face", src)
        wf["45"] = {
            "class_type": "MaskComposite",
            "inputs": {
                "destination": ["43", 0],
                "source": face,
                "x": 0,
                "y": 0,
                "operation": "subtract",
            },
        }
        part = ["45", 0]
        if keep_limbs:
            # Her arms and hands (grown) are never part of the change, so a hand on her cheek
            # or an arm across her chest stays (LIMB_WORDS).
            limbs = _sam(wf, "160", LIMB_WORDS[0], src)
            for node, word in zip(("162", "177"), LIMB_WORDS[1:], strict=True):
                wf[str(int(node) + 2)] = {
                    "class_type": "MaskComposite",
                    "inputs": {
                        "destination": limbs,
                        "source": _sam(wf, node, word, src),
                        "x": 0,
                        "y": 0,
                        "operation": "add",
                    },
                }
                limbs = [str(int(node) + 2), 0]
            wf["166"] = {
                "class_type": "GrowMask",
                "inputs": {"mask": limbs, "expand": LIMB_GROW, "tapered_corners": True},
            }
            wf["171"] = {
                "class_type": "GrowMask",
                "inputs": {"mask": limbs, "expand": PASTE_GROW, "tapered_corners": True},
            }
            wf["165"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": part,
                    "source": ["166", 0],
                    "x": 0,
                    "y": 0,
                    "operation": "subtract",
                },
            }
            part = ["165", 0]
            # What goes back from the photo afterwards (paste_held): her arms, hands and feet
            # (a little grown), face and hair.
            wf["175"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": ["171", 0],
                    "source": face,
                    "x": 0,
                    "y": 0,
                    "operation": "add",
                },
            }
            wf["176"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": ["175", 0],
                    "source": _sam(wf, "172", "hair", src),
                    "x": 0,
                    "y": 0,
                    "operation": "add",
                },
            }
            wf["168"] = {"class_type": "MaskToImage", "inputs": {"mask": ["176", 0]}}
            wf[HELD_NODE] = {"class_type": "PreviewImage", "inputs": {"images": ["168", 0]}}
        # The part to change, shown for keep_unchanged: there the edit's pixels are always
        # taken. A bigger butt in navy shorts differed too little from the old one, so half the
        # old shape came back beside the new (the owner's kneeling photo, 2026-11-08).
        wf["188"] = {"class_type": "MaskToImage", "inputs": {"mask": part}}
        wf[PART_NODE] = {"class_type": "PreviewImage", "inputs": {"images": ["188", 0]}}
        if hold_rest:
            # Everything but the part held at HOLD_BOOST, the part at 1: the pose stays (a
            # smaller bust stood up a woman lying on a lounger when the part was let go);
            # the change is milder ("Strong" lets the part go instead).
            wf["46"] = {"class_type": "InvertMask", "inputs": {"mask": part}}
            held = ["46", 0]
            if loras and not keep_limbs:
                # With a character LoRA her face is not held (left at 1), so it can become
                # hers: likeness after the face step up on 4 of 4 clothes edits (0.39 -> 0.54
                # on a big smiling face; PROVIDERS.md).
                wf["47"] = {
                    "class_type": "MaskComposite",
                    "inputs": {
                        "destination": held,
                        "source": face,
                        "x": 0,
                        "y": 0,
                        "operation": "subtract",
                    },
                }
                held = ["47", 0]
            patch |= {"ref_boost": HOLD_BOOST, "ref_boost_mask": held}
        else:
            patch |= {
                "ref_boost": REGION_BOOSTS.get(region, REGION_BOOST),
                "ref_boost_mask": part,
            }
    if keep_hands:
        # The hands of the photo (SAM 3.1), shown as a preview the studio fetches: the face
        # paste keeps them as they were (face_paste).
        hands = _sam(wf, "60", "hand", src)
        wf["63"] = {"class_type": "MaskToImage", "inputs": {"mask": hands}}
        wf[HANDS_NODE] = {"class_type": "PreviewImage", "inputs": {"images": ["63", 0]}}
    wf["13"] = {"class_type": "Krea2EditModelPatch", "inputs": patch}
    wf["4"] = {"class_type": "Krea2EditGroundedEncode", "inputs": encode}
    # CFG 1 (Turbo): the negative is not used, so it is not encoded.
    wf["5"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}}
    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["13", 0],
            "seed": seed,
            "steps": steps,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "denoise": 1.0,
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


# --- keep what the edit did not change ------------------------------------------------------
#
# The model redraws the whole photo at about 1 MP; where it changed nothing, the original's
# own pixels (at the original's resolution) are put back, so untouched parts stay exactly as
# they were. The change mask: per-pixel difference of the two (both lightly blurred, so a
# one-pixel shift or VAE texture is not a change), opened to drop thin lines and specks,
# grown and feathered. Thresholds from the studio's own runs (docs/PROVIDERS.md, Advanced).
KEEP_THRESHOLD = 0.07
KEEP_SPECK = 0.00015
KEEP_CLOSE = 41
KEEP_HOLE = 0.02
KEEP_GROW = 17
KEEP_FEATHER = 8.0
# Over this share changed (a new background), the edit is used whole: mixing in the few
# "unchanged" bits of the old scene left e.g. the white sky and a tree's edge behind a new
# white studio wall (seen on the studio's runs).
KEEP_MAX_CHANGE = 0.5


def change_mask(src: Any, out: Any) -> Any:
    """Where the edit changed the photo (PIL "L", the edit's size): the per-pixel difference
    of the two lightly blurred images over KEEP_THRESHOLD, specks dropped, closed, with small
    holes filled — a new beige beanie over blond hair left
    strands of the old hair where the colours were alike (seen on the owner's photos) —
    then grown and feathered."""
    import cv2
    import numpy as np
    from PIL import Image, ImageFilter

    w, h = out.size

    def soft(im: Image.Image) -> Any:
        return np.asarray(im.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255

    diff = np.abs(soft(src.resize((w, h), Image.LANCZOS)) - soft(out)).mean(axis=2)
    m = ((diff > KEEP_THRESHOLD) * 255).astype(np.uint8)
    # Specks (texture, a pixel's shift) go; thin new things stay: an opening dropped the wire
    # rims of new glasses and kept only their hinges (seen on the studio's runs).
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < KEEP_SPECK * w * h]
    if small:
        m[np.isin(labels, small)] = 0
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KEEP_CLOSE, KEEP_CLOSE))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk)
    # Holes inside a changed area are changed too, unless large (a whole person kept
    # inside a new background stays the original's).
    outer, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, outer, -1, 255, -1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.subtract(filled, m))
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < KEEP_HOLE * w * h:
            m[labels == i] = 255
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KEEP_GROW, KEEP_GROW))
    m = cv2.dilate(m, grow)
    return Image.fromarray(m, "L").filter(ImageFilter.GaussianBlur(KEEP_FEATHER))


def keep_unchanged(
    original: bytes, edited: bytes, changed_too: list[bytes] | None = None
) -> tuple[bytes, float]:
    """The edited photo with the unchanged parts taken from the original, as PNG, and the
    share of the picture the edit changed (0–1). `changed_too` (masks, white) are always taken
    from the edit (where the old hair was, HAIR_NODE; the part a body edit changed,
    PART_NODE)."""
    import io

    import numpy as np
    from PIL import Image, ImageChops, ImageFilter, ImageOps

    src = ImageOps.exif_transpose(Image.open(io.BytesIO(original))).convert("RGB")
    out = Image.open(io.BytesIO(edited)).convert("RGB")
    w, h = out.size
    mask = change_mask(src, out)
    for data in changed_too or []:
        extra = Image.open(io.BytesIO(data)).convert("L").resize((w, h), Image.BILINEAR)
        mask = ImageChops.lighter(mask, extra.filter(ImageFilter.GaussianBlur(KEEP_FEATHER)))
    changed = float(np.asarray(mask, dtype=np.float32).mean() / 255)
    # The original's size, but never smaller than the edit and at most 1.5x it (a huge upload
    # would make the redrawn parts look soft next to the untouched ones).
    long_src, long_out = max(src.size), max(w, h)
    scale = min(max(long_src, long_out), round(1.5 * long_out)) / long_src
    size = (round(src.width * scale), round(src.height * scale))
    if size != src.size:
        src = src.resize(size, Image.LANCZOS)
    if changed > KEEP_MAX_CHANGE:
        result = out.resize(size, Image.LANCZOS)
        buf = io.BytesIO()
        result.save(buf, "PNG")
        return buf.getvalue(), changed
    result = Image.composite(
        out.resize(size, Image.LANCZOS), src, mask.resize(size, Image.BILINEAR)
    )
    buf = io.BytesIO()
    result.save(buf, "PNG")
    return buf.getvalue(), changed


# --- the character's face, redrawn close up ----------------------------------------------
#
# In a whole-body photo the face is small, so the owner's LoRA has few pixels to work with.
# With a character LoRA, the main face (YuNet on the Mac, services/faces.py) is cropped as a
# square 2.2x its size, redrawn at 1024 px by Krea 2 Turbo image to image with the LoRA (1.25x)
# at denoise 0.5 (the instruction repeated, so the expression stays), and pasted back through
# a feathered ellipse. On six photos: likeness and sameness of her face across them up.
# Every face size (a big face kept too much of the photo's own face without it). Measured on
# the studio's runs: face likeness to the character's photos up (docs/PROVIDERS.md).
FACE_EXPAND = 2.2
FACE_PX = 1024
# 0.35: a smile kept on 6 of 6 seeds and closed eyes on 4 of 6 (at 0.4: 2 of 6, at 0.3: all),
# likeness 0.71 (0.4: 0.72, 0.3: 0.69; 10 faces x 2 seeds, 2026-11-07).
FACE_DENOISE = 0.35
# Her LoRA x1.5 in both parts of the face step (was 1.4 / 1.25): likeness 0.726 against 0.716
# and her face more alike across 12 photos (SFace between the results 0.60 against 0.56),
# laughs and closed eyes kept; a second face step or denoise 0.4 helped less (2026-11-08).
FACE_EDIT_LORA = 1.5
FACE_SWAP = (
    "Give her the face of {who}: her eyes, nose, lips, jaw and skin. Keep everything else in"
    " the photo exactly as it is, including anything on her face or in her hair, her makeup,"
    " her expression and her head angle."
)
# The character's LoRA is applied this much stronger in the face step (at most FACE_LORA_MAX).
FACE_LORA_BOOST = 1.5
FACE_LORA_MAX = 1.5
FACE_MAX_SHARE = 0.3
FACE_MIN_PX = 24
FACE_NOTE_MAX = 400


def face_square(
    box: tuple[float, float, float, float], width: int, height: int, any_size: bool = False
) -> tuple[int, int, int] | None:
    """(left, top, side) of the square to redraw around a face box, or None to leave it
    (too small, or already big unless any_size)."""
    x, y, w, h = box
    size = max(w, h)
    if size < FACE_MIN_PX or (not any_size and size > FACE_MAX_SHARE * min(width, height)):
        return None
    side = int(round(min(size * FACE_EXPAND, width, height)))
    cx, cy = x + w / 2, y + h / 2
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    return left, top, side


def upright(data: bytes) -> bytes:
    """The photo as PNG, turned the way its EXIF says (what people see)."""
    import io

    from PIL import Image, ImageOps

    buf = io.BytesIO()
    ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB").save(buf, "PNG")
    return buf.getvalue()


# Head: glasses, earrings, a hat, makeup. The head and shoulders (a square 3.5x the face,
# reaching down) are edited at 1024 px and only what changed is put back, so small
# accessories on a small person come out at their real size (measured, PROVIDERS.md).
HEAD_EXPAND = 3.5
HEAD_DOWN = 0.5
HEAD_NOTE = "Do not add hands."
HEAD_MAX_SHARE = 0.6


def head_square(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int] | None:
    x, y, w, h = box
    size = max(w, h)
    side = int(round(min(size * HEAD_EXPAND, width, height)))
    # A big face gains nothing from a close-up, and a square almost as wide as the photo cut
    # through the hair at its edge (the edit had moved the hair): edit the whole photo then.
    if size < FACE_MIN_PX or side > HEAD_MAX_SHARE * min(width, height):
        return None
    cx, cy = x + w / 2, y + h / 2 + HEAD_DOWN * h
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    return left, top, side


def crop_paste(
    data: bytes, edit: bytes, square: tuple[int, int, int], keep: bytes | None = None
) -> bytes:
    """Head: the edited square goes back where it changed something (change_mask), fading out
    at the square's edges; `keep` (hands) stays the photo's own."""
    import io

    from PIL import Image, ImageChops, ImageFilter, ImageOps

    left, top, side = square
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    patch = Image.open(io.BytesIO(edit)).convert("RGB")
    old = img.crop((left, top, left + side, top + side))
    mask = change_mask(old, patch).resize((side, side), Image.BILINEAR)
    border = max(4, side // 25)
    edge = Image.new("L", (side, side), 0)
    edge.paste(255, (border, border, side - border, side - border))
    mask = ImageChops.multiply(mask, edge.filter(ImageFilter.GaussianBlur(border / 2)))
    if keep:
        hands = Image.open(io.BytesIO(keep)).convert("L").resize((side, side), Image.BILINEAR)
        mask = ImageChops.subtract(mask, hands.filter(ImageFilter.MaxFilter(5)))
    img.paste(Image.composite(patch.resize((side, side), Image.LANCZOS), old, mask), (left, top))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def hold_face(original: bytes, result: bytes, box: tuple[float, float, float, float]) -> bytes:
    """Hair and Body: the face is put back exactly as it was (an oval inside the face box,
    feathered). Those edits re-render the face slightly even when it is held, and it came out
    smoother and a little different (seen on the studio's runs)."""
    import io

    from PIL import Image, ImageDraw, ImageFilter, ImageOps

    src = ImageOps.exif_transpose(Image.open(io.BytesIO(original))).convert("RGB")
    out = Image.open(io.BytesIO(result)).convert("RGB")
    sx, sy = out.width / src.width, out.height / src.height
    x, y, w, h = box
    oval = (
        x * sx + 0.08 * w * sx,
        y * sy - 0.05 * h * sy,
        (x + w) * sx - 0.08 * w * sx,
        (y + h) * sy + 0.02 * h * sy,
    )
    mask = Image.new("L", out.size, 0)
    ImageDraw.Draw(mask).ellipse(oval, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(2.0, 0.06 * w * sx)))
    out.paste(src.resize(out.size, Image.LANCZOS), (0, 0), mask)
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


HELD_NODE = "169"  # the body pass shows what it holds (keep_limbs) as a preview
HELD_FEATHER = 6.0


def paste_held(before: bytes, after: bytes, held: bytes) -> bytes:
    """The body pass: her arms, hands, face and hair (the held preview, white) exactly from
    `before` (feathered), the rest from `after`. Held at 4 in the edit, a hand on the cheek and
    the hair over the shoulder still went to show a much larger bust (2 of 2 seeds); taking
    everything outside the changed part from `before` left seams in the hair and half a bikini
    top where SAM took a selfie arm for buttocks."""
    import io

    from PIL import Image, ImageFilter

    out = Image.open(io.BytesIO(after)).convert("RGB")
    src = Image.open(io.BytesIO(before)).convert("RGB").resize(out.size, Image.LANCZOS)
    mask = Image.open(io.BytesIO(held)).convert("L").resize(out.size, Image.BILINEAR)
    out.paste(src, (0, 0), mask.filter(ImageFilter.GaussianBlur(HELD_FEATHER)))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def for_upload(data: bytes, longest: int) -> bytes:
    """The photo turned upright, at most `longest` px on its longer side, as a JPEG (95)."""
    import io

    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if max(img.size) > longest:
        img.thumbnail((longest, longest), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def face_crop(data: bytes, square: tuple[int, int, int]) -> bytes:
    import io

    from PIL import Image, ImageOps

    left, top, side = square
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    crop = img.crop((left, top, left + side, top + side)).resize((FACE_PX, FACE_PX), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "PNG")
    return buf.getvalue()


def face_paste(
    data: bytes,
    patch: bytes,
    square: tuple[int, int, int],
    keep: bytes | None = None,
    box: tuple[float, float, float, float] | None = None,
    grow: float = 0.0,
) -> bytes:
    """The redrawn face square pasted back. With `box` (the face's box in the photo) only a
    feathered oval around the face is taken: the whole square's ellipse also carried the
    face step's redraw of the background next to the head (a grey smudge in a beach sky);
    `grow` widens it (a share of the box, Face only: FACE_ONLY_GROW).
    `keep` (a mask of the square, white = the photo's own pixels, e.g. the hands SAM found)
    stays as it was, so a finger at the lips is never cut off by the new mouth."""
    import io

    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

    left, top, side = square
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    face = Image.open(io.BytesIO(patch)).convert("RGB").resize((side, side), Image.LANCZOS)
    mask = Image.new("L", (side, side), 0)
    if box:
        x, y, w, h = box
        x, y = x - left - grow * w / 2, y - top - grow * h / 2
        w, h = w * (1 + grow), h * (1 + grow)
        oval = (x - 0.12 * w, y - 0.18 * h, x + 1.12 * w, y + 1.1 * h)
        ImageDraw.Draw(mask).ellipse(oval, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(3.0, 0.08 * w)))
    else:
        pad = int(side * 0.12)
        ImageDraw.Draw(mask).ellipse((pad, pad, side - pad, side - pad), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(side * 0.06))
    if keep:
        hands = Image.open(io.BytesIO(keep)).convert("L").resize((side, side), Image.BILINEAR)
        hands = hands.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3))
        mask = ImageChops.subtract(mask, hands)
    img.paste(face, (left, top), mask)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# A face YuNet does not find as it is (a head tilted about 45 degrees: a selfie held
# sideways, a woman lying down) is found in the photo turned, the face step runs on the
# level face and the photo is turned back: on a lounger selfie the face step was skipped and
# her old face stayed (likeness 0.27; turned 0.62, 2026-11-07). The turn is the one YuNet is
# surest of: its score peaked at the level turn on 7 of 7 photos (its eye landmarks were off
# by 30+ degrees on tilted heads). A tilted face it does find works as well as turned (0.74
# vs 0.73, 0.67 vs 0.69), so that one stays as it is.
TURNS = (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0, 90.0, -90.0, 180.0)
# An upside-down face (text to image "lying on her back" with her head toward the camera) is
# found as it is too (0.70-0.85) but surer turned 180 (0.86-0.92); level faces scored higher as
# they are (0.94 vs 0.82-0.89). On it as it was, the face step left odd stripes and likeness
# 0.18-0.22 (2026-11-10).
FLIP_MARGIN = 0.04
# A face found only turned counts from this YuNet score: on back views (no face) it found
# "faces" in her soles and bottom at 0.60-0.70, and the face step drew one there (the owner's
# photo, 2026-11-10); her real faces found turned scored 0.865-0.946 on 24 test photos.
TURN_MIN_SCORE = 0.8


def turn(data: bytes, angle: float) -> bytes:
    """The photo turned by `angle` degrees (counter-clockwise), grown to keep every corner."""
    import io

    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    buf = io.BytesIO()
    img.rotate(angle, Image.BICUBIC, expand=True).save(buf, "PNG")
    return buf.getvalue()


def unturn_point(x: float, y: float, angle: float, size: tuple[int, int]) -> tuple[float, float]:
    """A point of the photo turned by `angle` (turn) where it is in the photo itself (size)."""
    import math

    w, h = size
    a = math.radians(angle)
    tw = abs(w * math.cos(a)) + abs(h * math.sin(a))
    th = abs(w * math.sin(a)) + abs(h * math.cos(a))
    dx, dy = x - tw / 2, y - th / 2
    return w / 2 + dx * math.cos(a) - dy * math.sin(a), h / 2 + dx * math.sin(a) + dy * math.cos(a)


def unturn(
    original: bytes, turned: bytes, angle: float, box: tuple[float, float, float, float]
) -> bytes:
    """The face pasted into the turned photo (`box` there) put back into the original: the
    turned photo is turned back, and only an oval around the face is taken from it."""
    import io

    from PIL import Image, ImageDraw, ImageFilter, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(original))).convert("RGB")
    work = Image.open(io.BytesIO(turned)).convert("RGB")
    x, y, w, h = box
    mask = Image.new("L", work.size, 0)
    ImageDraw.Draw(mask).ellipse((x - 0.2 * w, y - 0.28 * h, x + 1.2 * w, y + 1.2 * h), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(3.0, 0.08 * w)))
    back = work.rotate(-angle, Image.BICUBIC, expand=True)
    mask = mask.rotate(-angle, Image.BICUBIC, expand=True)
    left, top = (back.width - img.width) // 2, (back.height - img.height) // 2
    crop = (left, top, left + img.width, top + img.height)
    img.paste(back.crop(crop), (0, 0), mask.crop(crop))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# The face step's redraw keeps an open mouth: SAM 3.1 finds teeth and a tongue in the first
# part's result and that place is left out of the redraw (a noise mask). Without it the redraw
# closed a grin over the teeth (2 of 2 seeds) and pulled a tongue back in (1 of 2); with it
# both stayed, likeness 0.695 against 0.705 over 5 faces x 2 seeds (2026-11-08).
MOUTH_WORDS = ("teeth", "tongue")
MOUTH_GROW = 16


# The whole finished photo once more at 2K (Turbo + her LoRA, light image to image), so the face
# (redrawn close up at 1024 px) and the rest (the edit at about 1 MP) look alike: the owner saw
# a sharp face on a softer photo. At 0.12–0.25 text, patterns and objects stayed (a hoodie's
# letters, "MATCHA" on a cup, a leopard print) while hair and skin got detail; likeness 0.716
# against 0.714 before over 8 photos at 0.25 (2026-11-08).
DETAIL_DENOISE = 0.2
# Only her hair (SAM, grown, feathered) is redrawn, without her face (already redrawn
# close up; over it closed eyes opened) and hands (white nails turned nude); the rest keeps its
# pixels: over the whole photo a phone's clock "08:22" came out "W21" (and "lmxx2" with the
# trigger word in the text, which is therefore left out: likeness the same, 0.717 against 0.703
# over 4 photos); at 0.25 a key fob in her hand changed shape.
DETAIL_TEXT = "photo of a woman, sharp detailed photo, natural skin texture"
DETAIL_GROW = 8


# Text to image with her LoRA and words asking for a pose (asks_for_pose): the first
# COMPOSE_STEPS of Turbo's 8 run with her LoRA at COMPOSE_LORA, so the pose follows the words
# more, and at full strength for the rest (her face, body, skin). At full strength on every
# step (trained on mirror selfies) "leaning forward", "back to the camera" and "looking back"
# often became a standing mirror selfie; without her LoRA for those steps the poses were right
# but "completely nude" got underwear (2 of 9 nude) and looked staged; at 0.7 a bikini came
# back on "leaning forward"; at 0.85 the owner's 6 prompts stayed as natural and as nude as at
# full strength and leaning and sitting poses improved. Not for other words: a plain mirror
# selfie then had the phone over her face (2026-11-10, PROVIDERS.md › Advanced).
COMPOSE_STEPS = 2
COMPOSE_LORA = 0.85
COMPOSE_NODE = "13"


def compose_first(
    wf: dict[str, Any], steps: int = COMPOSE_STEPS, share: float = COMPOSE_LORA
) -> dict[str, Any]:
    """krea2.build_workflow's KSampler split in two with the same noise: the first `steps` on
    the model with the LoRAs at `share` of their strength, the rest at full strength."""
    first = wf.pop("7")["inputs"]
    model: list[Any] = ["1", 0]
    loaders = sorted(
        (k for k, n in wf.items() if n["class_type"] == "LoraLoaderModelOnly"), key=int
    )
    for index, node in enumerate(loaders):
        lora = wf[node]["inputs"]
        wf[str(30 + index)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model,
                "lora_name": lora["lora_name"],
                "strength_model": round(float(lora["strength_model"]) * share, 3),
            },
        }
        model = [str(30 + index), 0]
    common = {k: first[k] for k in ("cfg", "sampler_name", "scheduler", "positive", "negative")}
    common.update(noise_seed=first["seed"], steps=first["steps"])
    wf[COMPOSE_NODE] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            **common,
            "model": model,
            "add_noise": "enable",
            "start_at_step": 0,
            "end_at_step": steps,
            "return_with_leftover_noise": "enable",
            "latent_image": first["latent_image"],
        },
    }
    wf["7"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            **common,
            "model": first["model"],
            "add_noise": "disable",
            "start_at_step": steps,
            "end_at_step": 10000,
            "return_with_leftover_noise": "disable",
            "latent_image": [COMPOSE_NODE, 0],
        },
    }
    return wf


def detail_graph(
    *, image: str, width: int, height: int, lora: str, strength: float, seed: int
) -> dict[str, Any]:
    """The 2K detail pass (DETAIL_DENOISE) over the person in a finished photo."""
    wf: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": TURBO.filename, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": krea2.TEXT_ENCODER.filename,
                "type": krea2.FAMILY,
                "device": "default",
            },
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": krea2.VAE.filename}},
        "21": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": strength},
        },
        "10": {"class_type": "LoadImage", "inputs": {"image": image}},
        "101": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["10", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "center",
            },
        },
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["101", 0], "vae": ["3", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": DETAIL_TEXT}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["21", 0],
                "seed": seed,
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["233", 0],
                "denoise": DETAIL_DENOISE,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["237", 0], "filename_prefix": "ai-image-studio-advanced-detail"},
        },
        "40": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3.filename}},
    }
    # Her hair only, not her accessories: over her clothes it made smooth leggings blotchy,
    # over her skin it drew fine lines like stretch marks on a thigh and melted the charms of a
    # bracelet (the owner's photos, 2026-11-08).
    wf["257"] = {
        "class_type": "MaskComposite",
        "inputs": {
            "destination": _sam(wf, "253", "accessory", ["101", 0]),
            "source": _sam(wf, "255", "jewelry", ["101", 0]),
            "x": 0,
            "y": 0,
            "operation": "add",
        },
    }
    wf["231"] = {
        "class_type": "MaskComposite",
        "inputs": {
            "destination": _sam(wf, "250", "hair", ["101", 0]),
            "source": ["257", 0],
            "x": 0,
            "y": 0,
            "operation": "subtract",
        },
    }
    wf["232"] = {
        "class_type": "GrowMask",
        "inputs": {"mask": ["231", 0], "expand": DETAIL_GROW, "tapered_corners": True},
    }
    wf["240"] = {
        "class_type": "GrowMask",
        "inputs": {
            "mask": _sam(wf, "238", "hand", ["101", 0]),
            "expand": 12,
            "tapered_corners": True,
        },
    }
    wf["241"] = {
        "class_type": "MaskComposite",
        "inputs": {
            "destination": ["240", 0],
            "source": _sam(wf, "243", "face", ["101", 0]),
            "x": 0,
            "y": 0,
            "operation": "add",
        },
    }
    wf["242"] = {
        "class_type": "MaskComposite",
        "inputs": {
            "destination": ["232", 0],
            "source": ["241", 0],
            "x": 0,
            "y": 0,
            "operation": "subtract",
        },
    }
    wf["233"] = {
        "class_type": "SetLatentNoiseMask",
        "inputs": {"samples": ["12", 0], "mask": ["242", 0]},
    }
    wf["234"] = {"class_type": "MaskToImage", "inputs": {"mask": ["242", 0]}}
    wf["235"] = {
        "class_type": "ImageBlur",
        "inputs": {"image": ["234", 0], "blur_radius": 16, "sigma": 8.0},
    }
    wf["236"] = {"class_type": "ImageToMask", "inputs": {"image": ["235", 0], "channel": "red"}}
    wf["237"] = {
        "class_type": "ImageCompositeMasked",
        "inputs": {
            "destination": ["101", 0],
            "source": ["8", 0],
            "x": 0,
            "y": 0,
            "resize_source": False,
            "mask": ["236", 0],
        },
    }
    return wf


def face_graph(
    *,
    image: str,
    lora: str,
    strength: float,
    who: str,
    seed: int,
    note: str = "",
    keep_mouth: bool = False,
) -> dict[str, Any]:
    """The face step in one graph: the edit LoRA with the character LoRA (FACE_EDIT_LORA x)
    gives the crop her face while keeping what it shows — the expression (a laugh, closed
    eyes, a tongue out), makeup, things on the face; then Turbo with the character LoRA alone
    (FACE_LORA_BOOST x) redraws it lightly (FACE_DENOISE) from that latent for likeness. A
    plain image to image at denoise 0.5 erased laughs, closed eyes and tongues on the
    owner's photos (PROVIDERS.md). `note` (the user's instruction) steers the second part."""
    wf = build_edit(
        image=image,
        prompt=FACE_SWAP.format(who=who),
        width=FACE_PX,
        height=FACE_PX,
        seed=seed,
        edit_lora=DEFAULT_EDIT,
        loras=[(lora, min(FACE_LORA_MAX, strength * FACE_EDIT_LORA))],
        ref_boost=1.0,
        prefix="ai-image-studio-advanced-face",
    )
    text = f"close-up photo of {who}, natural skin texture"
    if note.strip():
        text = f"{text}. {note.strip()[:FACE_NOTE_MAX]}"
    wf["90"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": lora,
            "strength_model": min(FACE_LORA_MAX, strength * FACE_LORA_BOOST),
        },
    }
    wf["91"] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["2", 0]}}
    wf["92"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["91", 0]}}
    wf["93"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["90", 0],
            "seed": seed + 1,
            "steps": 8,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["91", 0],
            "negative": ["92", 0],
            "latent_image": ["7", 0],
            "denoise": FACE_DENOISE,
        },
    }
    if keep_mouth:
        wf["40"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3.filename}}
        wf["144"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
        teeth = _sam(wf, "145", MOUTH_WORDS[0], ["144", 0])
        tongue = _sam(wf, "147", MOUTH_WORDS[1], ["144", 0])
        wf["149"] = {
            "class_type": "MaskComposite",
            "inputs": {"destination": teeth, "source": tongue, "x": 0, "y": 0, "operation": "add"},
        }
        wf["150"] = {
            "class_type": "GrowMask",
            "inputs": {"mask": ["149", 0], "expand": MOUTH_GROW, "tapered_corners": True},
        }
        wf["151"] = {"class_type": "InvertMask", "inputs": {"mask": ["150", 0]}}
        wf["152"] = {
            "class_type": "SetLatentNoiseMask",
            "inputs": {"samples": ["7", 0], "mask": ["151", 0]},
        }
        wf["93"]["inputs"]["latent_image"] = ["152", 0]
    wf["8"]["inputs"]["samples"] = ["93", 0]
    return wf

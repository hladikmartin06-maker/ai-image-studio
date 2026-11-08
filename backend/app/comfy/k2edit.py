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


# A region edit (body shape: "breasts", "buttocks", "waist and hips"): SAM 3.1 (built-in
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
    "butt": ("buttocks", "hips"),
    "waist": ("waist", "hips"),
    "belly": ("belly", "waist"),
    "legs": ("legs", "thighs"),
    "arms": ("arms", "shoulders"),
    # The whole figure (slimmer, curvier, like the character): let go more gently (0.2), at
    # 0.05 it restaged a selfie.
    "figure": ("breasts", "buttocks", "waist", "hips", "thighs"),
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
    r"|turn\w*|walk\w*|jump\w*|danc\w*|hug\w*|cross\w*|rais\w*|lift\w*|hands?|arms?|legs?"
    r"|póz\w*|sed\w*|stoj\w*|lež\w*|ľah\w*|kľač\w*|otoč\w*|ruk\w*|nohy|nohu|nohami|drep\w*)\b",
    re.IGNORECASE,
)


def asks_for_pose(text: str) -> bool:
    return bool(POSE_WORDS.search(text))


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
REQUIRED_NODES = (*NODES, "EmptySD3LatentImage", "LoraLoaderModelOnly", "ImageScale")
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
        patch |= {"ref_boost": HOLD_BOOST, "ref_boost_mask": ["128", 0]}
    if region:
        # Only that part of the photo (SAM 3.1 finds it from the words, grown a little so a
        # bigger shape has room) is let go: the edit hardly looks at it there, and keeps the
        # rest as at ref_boost 1. Two words per part (BODY_PARTS), added together.
        words = HAIR_WORDS if region == "hair" else BODY_PARTS.get(region, (region,))
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
        if hold_rest:
            # Everything but the part held at HOLD_BOOST, the part at 1: the pose stays (a
            # smaller bust stood up a woman lying on a lounger when the part was let go);
            # the change is milder ("Strong" lets the part go instead).
            wf["46"] = {"class_type": "InvertMask", "inputs": {"mask": ["45", 0]}}
            held = ["46", 0]
            if loras:
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
                "ref_boost_mask": ["45", 0],
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


def keep_unchanged(original: bytes, edited: bytes) -> tuple[bytes, float]:
    """The edited photo with the unchanged parts taken from the original, as PNG, and the
    share of the picture the edit changed (0–1)."""
    import io

    import numpy as np
    from PIL import Image, ImageOps

    src = ImageOps.exif_transpose(Image.open(io.BytesIO(original))).convert("RGB")
    out = Image.open(io.BytesIO(edited)).convert("RGB")
    w, h = out.size
    mask = change_mask(src, out)
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
FACE_EDIT_LORA = 1.4
FACE_SWAP = (
    "Give her the face of {who}: her eyes, nose, lips, jaw and skin. Keep everything else in"
    " the photo exactly as it is, including anything on her face or in her hair, her makeup,"
    " her expression and her head angle."
)
# The character's LoRA is applied this much stronger in the face step (at most FACE_LORA_MAX).
FACE_LORA_BOOST = 1.25
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
TURNS = (-60.0, -45.0, -30.0, 30.0, 45.0, 60.0, 90.0, -90.0)


def turn(data: bytes, angle: float) -> bytes:
    """The photo turned by `angle` degrees (counter-clockwise), grown to keep every corner."""
    import io

    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    buf = io.BytesIO()
    img.rotate(angle, Image.BICUBIC, expand=True).save(buf, "PNG")
    return buf.getvalue()


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


def face_graph(
    *, image: str, lora: str, strength: float, who: str, seed: int, note: str = ""
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
    wf["8"]["inputs"]["samples"] = ["93", 0]
    return wf

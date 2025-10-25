from __future__ import absolute_import, division, unicode_literals, print_function

import os, csv
import numpy as np
from torch.utils.data import Dataset
from dataloaders.rawvideo_util import RawVideoExtractor


class Custom_DataLoader(Dataset):
    """
    스키마: class, video_id, sentence, path
    - data_path: train.csv / val.csv / test.csv 가 들어있는 디렉토리
    - subset: "train" | "val" | "test"  (해당 csv 반드시 존재)
    - features_path: CSV의 'path'가 상대경로일 때만 보정에 사용
    반환 형식(LSMDC 호환):
        __getitem__ -> (pairs_text, pairs_mask, pairs_segment, video, video_mask)
    """

    def __init__(
        self,
        subset,                 # "train" | "val" | "test"
        data_path,              # 디렉토리: {subset}.csv 필수
        features_path,          # 상대 경로 보정 용도 (절대경로면 무시)
        tokenizer,
        max_words=30,
        feature_framerate=1.0,
        max_frames=100,
        image_resolution=224,
        frame_order=0,          # 0: normal, 1: reverse, 2: random
        slice_framepos=2,       # 0: head, 1: tail, 2: uniform
    ):
        assert subset in ["train", "val", "test"]
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"[custom] data_path는 디렉토리여야 합니다: {data_path}")

        csv_path = os.path.join(data_path, f"{subset}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[custom] {csv_path} 가 없습니다. ({subset}.csv 필수)")

        self.subset = subset
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.feature_framerate = feature_framerate
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        # --- CSV 파싱 (엄격)
        samples, n_missing = [], 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fns = reader.fieldnames or []
            need = ["class", "video_id", "sentence", "path"]
            missing_cols = [k for k in need if k not in fns]
            if missing_cols:
                raise RuntimeError(f"[custom] {csv_path} 헤더에 {missing_cols} 없음. 현재 헤더: {fns}")

            for row in reader:
                vid = row["video_id"]
                cap = row["sentence"]
                vpath = row["path"]

                if not vid or not cap or not vpath:
                    n_missing += 1
                    continue

                # 상대경로 → features_path 기준 보정
                if not os.path.isabs(vpath) and os.path.isdir(features_path):
                    vpath = os.path.normpath(os.path.join(features_path, vpath))

                if not os.path.exists(vpath):
                    n_missing += 1
                    continue

                samples.append((vid, cap, vpath))

        if len(samples) == 0:
            raise RuntimeError(f"[custom] 유효 샘플이 없습니다: {csv_path}")
        if n_missing > 0:
            print(f"[custom] 경고: 누락/무효 샘플 {n_missing}개는 건너뜀. ({csv_path})")

        self.samples = {i: s for i, s in enumerate(samples)}

        # 비디오 디코더
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution
        )
        # LSMDC 호환 토큰
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

    def __len__(self):
        return len(self.samples)

    # ----- 텍스트 토크나이즈 (LSMDC 포맷과 동일)
    def _get_text(self, caption):
        k = 1
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        words = self.tokenizer.tokenize(caption)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_len = self.max_words - 1
        if len(words) > total_len:
            words = words[:total_len]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)
        pairs_segment[0] = np.array(segment_ids)
        return pairs_text, pairs_mask, pairs_segment

    # ----- 비디오 로드 (LSMDC 포맷과 동일)
    def _get_rawvideo(self, video_path):
        video_mask = np.zeros((1, self.max_frames), dtype=np.int64)
        video = np.zeros(
            (1, self.max_frames, 1, 3,
             self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=np.float32
        )

        raw = self.rawVideoExtractor.get_video_data(video_path)['video']
        if len(raw.shape) <= 3:
            return video, video_mask

        raw_slice = self.rawVideoExtractor.process_raw_data(raw)  # (L, T, 3, H, W) or (L, 3, H, W)

        if self.max_frames < raw_slice.shape[0]:
            if self.slice_framepos == 0:
                video_slice = raw_slice[:self.max_frames, ...]
            elif self.slice_framepos == 1:
                video_slice = raw_slice[-self.max_frames:, ...]
            else:
                idx = np.linspace(0, raw_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_slice[idx, ...]
        else:
            video_slice = raw_slice

        video_slice = self.rawVideoExtractor.process_frame_order(
            video_slice, frame_order=self.frame_order
        )

        L = video_slice.shape[0]
        if L > 0:
            video[0][:L, ...] = video_slice
            video_mask[0][:L] = 1

        return video, video_mask

    def __getitem__(self, index):
        _vid, caption, vpath = self.samples[index]
        pairs_text, pairs_mask, pairs_segment = self._get_text(caption)
        video, video_mask = self._get_rawvideo(vpath)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask
import torch
from PIL import Image
from typing import List
from abc import ABC, abstractmethod
from transformers import AutoModel
from transformers import ColPaliForRetrieval, ColPaliProcessor

class BaseEmbeddingModel(ABC):
    def __init__(self, device: str = "auto"):
        self.device = device
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load specific model weights and processors."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> torch.Tensor:
        """Process text and return standardized result."""
        pass

    @abstractmethod
    def embed_image(self, image_paths: List[str]) -> List[torch.Tensor]:
        """Process image and return standardized result."""
        pass

class JinaV4Model(BaseEmbeddingModel):
    def __init__(self, device = "auto", multivector=False):
        self.model_name = "jinaai/jina-embeddings-v4"
        super().__init__(device)
        self.multivector = multivector

    def load_model(self):
        print(f"Loading Jina v4: {self.model_name}...")
        self._model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=self.device
        )

    @torch.no_grad()
    def embed_image(self, image_paths: List, batch_size: int) -> List[torch.Tensor]:
        return self._model.encode_image(
            images=image_paths,
            task="retrieval",
            batch_size=batch_size,
            return_multivector=self.multivector
        )

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:    
        return self._model.encode_text(
            texts=text,
            task="retrieval",
            prompt_name = "query",
            return_multivector=self.multivector
        )

class ColPaliModel(BaseEmbeddingModel):
    def __init__(self, device = "auto"):
        self.model_name = "vidore/colpali-v1.3-hf"
        super().__init__(device)

    def load_model(self):
        print(f"Loading ColPali: {self.model_name}...")
        self._model = ColPaliForRetrieval.from_pretrained(
            self.model_name, 
            dtype=torch.bfloat16,
            device_map=self.device
        )
        self._processor = ColPaliProcessor.from_pretrained(
            self.model_name, 
            use_fast=True
        )

    @torch.no_grad()
    def embed_image(self, image_paths: List, batch_size: int) -> List[torch.Tensor]:
        batch_images = self._processor(
            images=[Image.open(path).convert("RGB") 
                for path in image_paths], 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        return self._model(**batch_images).embeddings

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor: 
        query = self._processor(
            text=[text],
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        return self._model(**query).embeddings[0].cpu().tolist()
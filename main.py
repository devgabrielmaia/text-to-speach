"""Converte um arquivo Markdown em áudio usando a API de TTS da OpenAI.

Uso:
    uv run main.py docs/guia.md
    uv run main.py docs/guia.md -o meu_audio
    uv run main.py docs/guia.md -v nova --instructions "Fale de forma animada"
"""

import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

OUTPUT_DIR = Path(__file__).parent / "storage" / "audios"
DEFAULT_MODEL = "gpt-4o-mini-tts"
MAX_CHARS = 4000  # limite seguro por requisição; textos maiores são divididos
VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]


def markdown_to_text(md: str) -> str:
    """Remove a sintaxe do Markdown para que ela não seja lida em voz alta."""
    text = re.sub(r"```.*?```", "\n(Trecho de código omitido.)\n", md, flags=re.DOTALL)  # blocos de código
    text = re.sub(r"^\s*([-*_]\s*){3,}$", "", text, flags=re.MULTILINE)  # linhas horizontais
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # comentários HTML
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)  # tags HTML
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # imagens
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links -> só o texto
    text = re.sub(r"<?https?://[^\s>)]+>?", "", text)  # URLs soltas
    text = re.sub(r"`([^`]+)`", r"\1", text)  # código inline
    text = re.sub(r"(\*\*|\*|~~)(\S.*?\S|\S)\1", r"\2", text)  # negrito/itálico/riscado
    text = re.sub(r"(?<!\w)(__|_)(\S.*?\S|\S)\1(?!\w)", r"\2", text)  # _itálico_ (preserva snake_case)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"\|?[\s:|-]*-{3,}[\s:|-]*", line) or re.fullmatch(r"([-*_]\s*){3,}", line):
            continue  # separador de tabela ou linha horizontal
        if line.startswith("|"):  # linha de tabela -> células separadas por vírgula
            line = ", ".join(c.strip() for c in line.strip("|").split("|") if c.strip())
        line = re.sub(r"^#{1,6}\s*", "", line)  # títulos
        line = re.sub(r"^>\s?", "", line)  # citações
        line = re.sub(r"^([-*+]|\d+\.)\s+(\[[ xX]\]\s+)?", "", line)  # listas e checkboxes
        line = re.sub(r"\s{2,}", " ", line)
        line = re.sub(r"\s+([.,;:!?])", r"\1", line)  # espaço órfão antes de pontuação
        if line and line[-1] not in ".!?:;…,)":
            line += "."  # pausa ao fim de títulos, itens de lista e linhas de tabela
        lines.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Divide o texto em pedaços de até max_chars, quebrando entre frases."""
    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > max_chars:  # frase gigante sem pontuação
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def text_to_audio(
    text: str,
    output_name: str,
    voice: str = "coral",
    model: str = DEFAULT_MODEL,
    instructions: str | None = None,
    speed: float = 1.0,
) -> Path:
    client = OpenAI()  # lê OPENAI_API_KEY do ambiente/.env
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{output_name}.mp3"

    extra = {"instructions": instructions} if instructions else {}
    with output_path.open("wb") as f:
        for chunk in split_text(text):
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=chunk,
                response_format="mp3",
                speed=speed,
                **extra,
            )
            f.write(response.content)
    return output_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Converte um arquivo Markdown em áudio (MP3) com a API da OpenAI.")
    parser.add_argument("file", type=Path, help="Arquivo .md a ser convertido")
    parser.add_argument("-o", "--output", help="Nome do arquivo de saída sem extensão (padrão: nome do .md)")
    parser.add_argument("-v", "--voice", default="coral", choices=VOICES, help="Voz (padrão: coral)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Modelo (padrão: {DEFAULT_MODEL})")
    parser.add_argument("-i", "--instructions", help='Instruções de estilo, ex.: "Fale devagar e com calma"')
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Velocidade, de 0.25 a 4.0 (padrão: 1.0)")
    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(f"arquivo não encontrado: {args.file}")

    text = markdown_to_text(args.file.read_text(encoding="utf-8"))
    if not text:
        sys.exit("Erro: o arquivo não tem texto para narrar.")

    output_name = args.output or args.file.stem
    path = text_to_audio(text, output_name, args.voice, args.model, args.instructions, args.speed)
    print(f"Áudio salvo em: {path}")


if __name__ == "__main__":
    main()

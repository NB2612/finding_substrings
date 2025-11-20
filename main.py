import argparse
from typing import List, Union
from termcolor import colored
from itertools import cycle

# Импортируем ранее реализованные функции поиска
# from search_module import search  # если код search.py в отдельном модуле

def search(text: str, sub_string: Union[str, List[str]], case_sensitivity=True, method="first", count=None, algorithm="ak"):
    from search import search
    return search(text, sub_string, case_sensitivity, method, count, algorithm)


def highlight_text(text: str, matches: dict[str, tuple[int, ...]]) -> str:
    """
    Возвращает текст с цветовым выделением найденных подстрок.
    Каждая подстрока выделяется своим цветом.
    """
    if not matches:
        return text

    colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']
    color_cycle = cycle(colors)

    # Формируем список всех найденных вхождений
    highlights = []
    for sub, indices in matches.items():
        if indices:
            color = next(color_cycle)
            for idx in indices:
                highlights.append((idx, idx + len(sub), sub, color))

    # Сортируем по индексу начала
    highlights.sort(key=lambda x: x[0])

    result_text = ""
    last_idx = 0
    for start, end, sub, color in highlights:
        # добавляем текст до выделения
        result_text += text[last_idx:start]
        # добавляем цветной подстроку
        result_text += colored(text[start:end], color)
        last_idx = end
    # добавляем остаток текста
    result_text += text[last_idx:]
    return result_text


def main():
    global text_lines
    parser = argparse.ArgumentParser(description="Поиск подстрок в тексте с подсветкой.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--text", type=str, help="Целевая строка для поиска")
    group.add_argument("-f", "--file", type=str, help="Путь до файла для поиска")

    parser.add_argument("-s", "--substrings", type=str, nargs="+",
                        required=True, help="Подстроки для поиска")
    parser.add_argument("-a", "--algorithm", type=str,
                        choices=["kmp", "bm", "ak"], default="ak", help="Алгоритм поиска")
    parser.add_argument("-c", "--case", action="store_true", help="Учитывать регистр")
    parser.add_argument("-m", "--method", type=str, choices=["first", "last"],
                        default="first", help="Метод поиска (first/last)")
    parser.add_argument("-n", "--count", type=int, default=None, help="Максимальное количество совпадений")
    parser.add_argument("-l", "--lines", type=int, default=10,
                        help="Максимальное количество строк для вывода")

    args = parser.parse_args()

    # Получаем текст
    if args.text:
        text_lines = args.text.splitlines()
    else:
        with open(args.file, encoding="utf-8") as f:
            text_lines = f.readlines()

    # Ограничение по количеству выводимых строк
    text_lines = text_lines[:args.lines]

    # Объединяем в один текст для поиска
    text = "\n".join(text_lines)

    # Выполняем поиск
    result = search(
        text=text,
        sub_string=args.substrings,
        case_sensitivity=args.case,
        method=args.method,
        count=args.count,
        algorithm=args.algorithm
    )

    # Если результат кортеж (одна подстрока), приводим к словарю для подсветки
    if isinstance(result, tuple):
        result_dict = {args.substrings[0]: result}
    else:
        result_dict = result

    # Выводим результат с подсветкой
    highlighted = highlight_text(text, result_dict)
    print(highlighted)


if __name__ == "__main__":
    main()

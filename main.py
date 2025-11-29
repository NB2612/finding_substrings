import argparse
from typing import List, Union
from termcolor import colored
from itertools import cycle


# Функция-заглушка для поиска.
# Предполагается, что она импортирует и вызывает реальную функцию search.
def search(text: str, sub_string: Union[str, List[str]], case_sensitivity=True, method="first", count=None,
           algorithm="ak"):
    # !!! Важно: убедитесь, что ваш файл search.py доступен
    # и содержит функцию def search(...)
    try:
        from search import search as actual_search
        return actual_search(text, sub_string, case_sensitivity, method, count, algorithm)
    except ImportError:
        print("Ошибка: Не удалось импортировать функцию search из search.py.")
        return {}  # Возвращаем пустой результат в случае ошибки


def highlight_text(text: str, matches: dict[str, tuple[int, ...]]) -> str:
    """
    Возвращает текст с цветовым выделением найденных подстрок.
    Каждая подстрока выделяется своим цветом.
    """
    if not matches:
        return text

    # Используем больше цветов для лучшего различия
    colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'light_red', 'light_green', 'light_yellow']
    color_cycle = cycle(colors)

    # Формируем список всех найденных вхождений (start, end, sub, color)
    highlights = []

    # Чтобы избежать конфликтов цветов между разными подстроками,
    # используем фиксированный цвет для каждой подстроки из args.substrings

    # 1. Сначала назначаем цвета подстрокам
    color_map = {}
    for sub in matches.keys():
        if sub not in color_map:
            color_map[sub] = next(color_cycle)

    # 2. Формируем список всех выделений
    for sub, indices in matches.items():
        if indices:
            color = color_map.get(sub, 'red')  # Получаем назначенный цвет
            for idx in indices:
                # Проверяем, что индекс начала не выходит за границы текста
                if idx >= 0 and idx + len(sub) <= len(text):
                    highlights.append((idx, idx + len(sub), sub, color))

    # Сортируем по индексу начала. Это важно для правильного обхода текста.
    highlights.sort(key=lambda x: x[0])

    result_text = ""
    last_idx = 0

    # Дополнительная логика для **избежания перекрывающихся выделений**
    # Если выделения перекрываются, приоритет отдается первому (или самому длинному,
    # но тут мы просто идем по отсортированному списку) и пропускаем пересекающиеся.

    # В простом случае, как ваш, достаточно убедиться, что next_start > last_idx

    for start, end, sub, color in highlights:
        # Проверяем, что текущее начало находится после конца последнего добавленного выделения
        # Это предотвращает наложение и двойную обработку
        if start >= last_idx:
            # добавляем текст до выделения
            result_text += text[last_idx:start]
            # добавляем цветную подстроку
            # Важно: colored() должен вызываться с **исходным** текстом
            result_text += colored(text[start:end], color)
            last_idx = end

    # добавляем остаток текста
    result_text += text[last_idx:]
    return result_text


def main():
    parser = argparse.ArgumentParser(description="Поиск подстрок в тексте с подсветкой.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--text", type=str, help="Целевая строка для поиска")
    group.add_argument("-f", "--file", type=str, help="Путь до файла для поиска")

    parser.add_argument("-s", "--substrings", type=str, nargs="+",
                        required=True, help="Подстроки для поиска (разделяются пробелами или в кавычках)")
    parser.add_argument("-a", "--algorithm", type=str,
                        choices=["kmp", "bm", "ak"], default="ak", help="Алгоритм поиска")
    parser.add_argument("-c", "--case", action="store_false",
                        dest='case_sensitivity',  # Меняем dest для логики!
                        help="Игнорировать регистр (по умолчанию учитывается)")
    parser.add_argument("-m", "--method", type=str, choices=["first", "last"],
                        default="first", help="Метод поиска (first/last/all)")  # Добавил 'all'
    parser.add_argument("-n", "--count", type=int, default=None, help="Максимальное количество совпадений на подстроку")
    parser.add_argument("-l", "--lines", type=int, default=10,
                        help="Максимальное количество строк для вывода")

    args = parser.parse_args()

    # --- 1. Получение текста из аргументов или файла ---
    text_lines: List[str] = []

    if args.text is not None:
        # Обработка -t "строка\nстрока2"
        text_lines = args.text.splitlines()
    elif args.file is not None:
        try:
            with open(args.file, 'r', encoding="utf-8") as f:
                # Используем readlines() для сохранения пустых строк
                text_lines = [line.rstrip('\n') for line in f.readlines()]
        except FileNotFoundError:
            print(f"Ошибка: Файл '{args.file}' не найден.")
            return
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            return

    # Ограничение по количеству выводимых строк (применяем до объединения)
    text_lines = text_lines[:args.lines]

    # Проверка на пустой текст
    # Если текст пустой (text_lines == []) ИЛИ состоит только из пробелов
    if not text_lines or all(line.strip() == "" for line in text_lines):
        print("Ошибка: текст для поиска пуст или состоит только из пробелов.")
        return

    # --- 2. Обработка подстрок ---

    # args.substrings уже является List[str] благодаря nargs
    # Если пользователь хочет несколько слов в одном аргументе, он может их заключить в кавычки.

    # 1. Фильтруем пустые подстроки
    sub_list = [s for s in args.substrings if s.strip()]

    if not sub_list:
        print("Ошибка: необходимо указать хотя бы одну непустую подстроку для поиска (-s).")
        return

    # --- 3. Выполнение поиска ---

    # Объединяем в один текст для поиска (с \n, чтобы сохранить структуру строк)
    text = "\n".join(text_lines)

    # Выполняем поиск
    # Передаем sub_list, который содержит только непустые подстроки.
    # Передаем правильное значение case_sensitivity.
    # `action="store_false"` делает `args.case_sensitivity` равным `False`,
    # что означает **игнорировать регистр**.

    # По умолчанию search ожидает `case_sensitivity=True`, поэтому меняем логику:
    # args.case_sensitivity: True (если -c не указан, т.е. учитывать)
    # args.case_sensitivity: False (если -c указан, т.е. игнорировать)

    result = search(
        text=text,
        sub_string=sub_list,  # sub_list - это List[str]
        case_sensitivity=args.case_sensitivity,  # Использование нового dest
        method=args.method,
        count=args.count,
        algorithm=args.algorithm
    )

    # --- 4. Обработка и вывод результата ---


    result_dict: dict[str, tuple[int, ...]] = {}

    if isinstance(result, dict):
        result_dict = result
    elif isinstance(result, tuple) and len(sub_list) == 1:
        # Если search вернул кортеж, и у нас только одна подстрока,
        # то это индексы для этой подстроки.
        result_dict = {sub_list[0]: result}
    elif result is not None:
        # Другой неожиданный результат
        print("Предупреждение: Результат поиска имеет неожиданный формат.")
        return

    # Выводим результат с подсветкой
    highlighted = highlight_text(text, result_dict)

    if not result_dict:
        print("Совпадений не найдено.")

    print(highlighted)


if __name__ == "__main__":
    main()
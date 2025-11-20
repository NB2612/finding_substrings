"""
Модуль для поиска вхождений подстроки(ок) в строке с использованием
алгоритма Бойера-Мура.

Функция search поддерживает:
- Чувствительность/нечувствительность к регистру.
- Поиск с начала ('first') или с конца ('last').
- Ограничение на количество первых/последних вхождений (count).
- Поиск одной подстроки или нескольких подстрок.
"""
from collections import deque
from typing import Union, Optional, List, Callable
import time

def log_time_decorator(func):
    """Декоратор для логирования времени выполнения функции."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(
            f"[Декоратор] {func.__name__} выполнена за {end - start:.6f} секунд")
        return result

    return wrapper

SUPPORT_ALG = {'kmp' , 'bm', 'ak'}


# ------------------- AHO-CORASICK -------------------
class ACNode:
    __slots__ = ['children', 'fail', 'output']

    def __init__(self):
        """
        Инициализация узла Ахо-Корасик.
        - children: словарь переходов по символам в дочерние узлы
        - fail: ссылка на узел для "отката" при несовпадении
        - output: список паттернов, которые оканчиваются в этом узле
        """
        self.children: dict[str, ACNode] = {}
        self.fail: Optional[ACNode] = None
        self.output: List[str] = []

    def __repr__(self) -> str:
        """Строковое представление узла для отладки."""
        return f"AhoCorasickNode(output={self.output})"

    def add_pattern(self, pattern: str) -> None:
        """
        Добавляет шаблон в бор (Trie), начиная с текущего узла.

        :param pattern: Строка-паттерн для добавления
        """
        node = self
        for char in pattern:
            if char not in node.children:
                node.children[char] = ACNode()
            node = node.children[char]
        node.output.append(pattern)

    def build_failure_links(self) -> None:
        """
        Строит суффиксные ссылки (fail links) для всех узлов в Trie.
        Используется для ускорения поиска нескольких паттернов.
        """
        queue = deque()

        # Начальные дети корня получают fail-ссылку на корень
        for child in self.children.values():
            child.fail = self
            queue.append(child)

        # BFS по дереву для построения fail-ссылок
        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                queue.append(child)
                fail_node = current.fail
                while fail_node is not self and char not in fail_node.children:
                    fail_node = fail_node.fail
                child.fail = fail_node.children.get(char, self)
                # Наследуем все паттерны fail-узла
                child.output += child.fail.output

    def search_in(self, text: str, limit: Optional[int] = None) -> dict[str, List[int]]:
        """
        Выполняет поиск всех добавленных шаблонов в заданном тексте.

        :param text: Текст для поиска
        :param limit: Максимальное количество совпадений, None = без ограничений
        :return: Словарь {паттерн: список индексов начала вхождений}
        """
        # Собираем все паттерны, добавленные в Trie
        all_patterns = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if node.output:
                all_patterns.update(node.output)
            stack.extend(node.children.values())

        result = {pattern: [] for pattern in all_patterns}

        node = self
        total_found = 0

        # Проходим по тексту символ за символом
        for i, char in enumerate(text):
            # Идем по fail-ссылкам, если нет перехода по символу
            while node is not self and char not in node.children:
                node = node.fail

            # Делаем переход, если возможно
            if char in node.children:
                node = node.children[char]
            else:
                node = self

            # Для каждого паттерна, завершившегося в текущем узле
            for pattern in node.output:
                if limit is not None and total_found >= limit:
                    return result
                start_index = i - len(pattern) + 1
                result[pattern].append(start_index)
                total_found += 1
                if limit is not None and total_found >= limit:
                    return result
        return result


def _ak_search(text_to_search: str,
               patterns: List[str],
               limit: Optional[int] = None
               ) -> dict[str, List[int]]:
    """
    Основная функция поиска с использованием алгоритма Ахо-Корасик.

    :param text_to_search: Текст, в котором ищем
    :param patterns: Список паттернов для поиска
    :param limit: Максимальное количество совпадений на паттерн
    :return: Словарь {паттерн: список индексов начала совпадений}
    """
    if not patterns or not text_to_search or (limit is not None and limit <= 0):
        return {p: [] for p in patterns if p}

    non_empty_patterns = [p for p in patterns if p]
    if not non_empty_patterns:
        return {p: [] for p in patterns if p}

    # Строим Trie и fail-ссылки
    root = ACNode()
    for pattern in non_empty_patterns:
        root.add_pattern(pattern)
    root.build_failure_links()

    # Выполняем поиск в тексте
    raw_result = root.search_in(text_to_search, limit)

    # Сохраняем исходный порядок паттернов
    final_result = {}
    for pattern in patterns:
        final_result[pattern] = raw_result.get(pattern, []) if pattern else []

    return final_result


def _process_ak_last(text_to_search: str,
                     sub_list: List[str],
                     limit_res: Optional[int] = None) -> List[tuple[int, str]]:
    """
    Обрабатывает поиск с методом 'last', используя алгоритм Ахо-Корасик.
    Поиск выполняется в обратном порядке текста и паттернов.

    :param text_to_search: Исходный текст
    :param sub_list: Список паттернов
    :param limit_res: Максимальное количество совпадений
    :return: Список кортежей (индекс начала, паттерн)
    """
    # Переворачиваем паттерны и текст для поиска с конца
    rev_sub_list = [pattern[::-1] for pattern in sub_list]
    string_rev = text_to_search[::-1]

    raw = _ak_search(string_rev, rev_sub_list, limit_res)
    all_matches = []
    total = 0

    # Преобразуем индексы обратно в оригинальный текст
    for pattern_rev, indices in raw.items():
        if limit_res is not None and total >= limit_res:
            break
        orig_pattern = sub_list[rev_sub_list.index(pattern_rev)]
        for rev_idx in indices:
            if limit_res is not None and total >= limit_res:
                break
            orig_idx = len(text_to_search) - rev_idx - len(orig_pattern)
            all_matches.append((orig_idx, orig_pattern))
            total += 1

    return all_matches


# ------------------- END AHO-CORASICK -------------------

def _kmp_search(text: str, pattern: str,
                max_results: Optional[int] = None) -> List[int]:
    """
    Поиск подстроки в строке с использованием алгоритма Кнута–Морриса–Пратта (KMP).

    Алгоритм использует префикс-функцию (LPS — longest prefix suffix),
    чтобы избежать повторных сравнений при несовпадениях, что обеспечивает
    линейную асимптотику O(n + m), где:
        n — длина текста,
        m — длина шаблона.

    Параметры:
        text (str): строка, в которой ведётся поиск.
        pattern (str): подстрока, которую требуется найти.
        max_results (Optional[int]): максимальное количество вхождений.
            Если None — допустимо неограниченное число вхождений.

    Возвращает:
        List[int]: список индексов начала найденных подстрок в `text`.
    """

    if not pattern:
        return []

    # Построение таблицы LPS (длины наибольшего суффикса-префикса)
    lps = [0] * len(pattern)
    j = 0  # длина текущего совпавшего префикса
    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j

    results = []
    i = j = 0  # i — индекс в text, j — индекс в pattern

    # Основной цикл поиска
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1

            # Найдено совпадение целого шаблона
            if j == len(pattern):
                results.append(i - j)

                # Ограничение на количество результатов
                if max_results is not None and len(results) >= max_results:
                    return results

                j = lps[j - 1]  # переход к следующему возможному совпадению

        else:
            # Если несовпадение — откатываемся по lps
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return results

def _bm_search(text_to_search: str,
               sub_list: List[str],
               limit: Optional[int] = None
) -> List[tuple[int, str]]:
    """
       Реализация алгоритма Бойера-Мура для поиска одной подстроки в тексте.

       :param text_to_search: Текст, в котором ищем
       :param sub_list: Паттерн для поиска
       :param limit: Максимальное количество совпадений (None = без ограничений)
       :return: Список индексов начала совпадений
       """
    if not sub_list or not text_to_search:
        return []

    m = len(sub_list)
    n = len(text_to_search)
    if m > n:
        return []

    # ------------------- Preprocessing: bad character rule -------------------
    bad_char = {}
    for i, char in enumerate(sub_list):
        bad_char[char] = i  # Последнее вхождение символа в паттерн

    # ------------------- Search -------------------
    results = []
    s = 0  # смещение относительно текста
    limit_val = limit if limit is not None else float('inf')

    while s <= n - m:
        j = m - 1

        # Сравнение паттерна с текстом справа налево
        while j >= 0 and sub_list[j] == text_to_search[s + j]:
            j -= 1

        if j < 0:
            # Полное совпадение найдено
            results.append(s)
            if len(results) >= limit_val:
                break
            # Сдвигаем по правилу "good suffix" или на 1
            s += 1 if s + m >= n else m - bad_char.get(text_to_search[s + m], -1)
        else:
            # Сдвиг по правилу плохого символа
            char_shift = j - bad_char.get(text_to_search[s + j], -1)
            s += max(1, char_shift)

    return results


def _process_first_method(alg: str,
                          text_to_search: str,
                          sub_list: List[str],
                          count: Optional[int]
) -> List[tuple[int, str]]:
    """
       Выполняет поиск по тексту, возвращая первые найденные вхождения подстрок
       в соответствии с выбранным алгоритмом.

       Функция обрабатывает алгоритмы поиска по двум вариантам:
       - "ak": алгоритм, который принимает список подстрок и возвращает словарь
         {подстрока -> [позиции]}.
       - "kmp" и "bm": алгоритмы, которые принимают одиночную подстроку
         и возвращают список позиций.

       Результат всегда представляет собой список кортежей вида:
           (позиция_в_тексте, подстрока)
       при этом список отсортирован по позиции появления подстроки в тексте.

       Ограничение по количеству найденных результатов задаётся параметром `count`.
       Если `count` равен `None`, возвращаются все найденные вхождения.
       Если ограничение достигнуто — дальнейший поиск не выполняется.

       Args:
           alg (str): Название алгоритма поиска.
                      Допустимые значения: "ak", "kmp", "bm".
           text_to_search (str): Исходный текст, в котором осуществляется поиск.
           sub_list (List[str]): Список подстрок, подготовленных для поиска
                                 (с учётом регистра и без пустых значений).
           count (Optional[int]): Максимальное количество совпадений, которое
                                  нужно найти. `None` означает без ограничений.

       Returns:
           List[tuple[int, str]]: Список найденных совпадений.
               Каждый элемент — кортеж (индекс, подстрока).
               Порядок результата — по возрастанию индекса.
       """
    all_matches = []

    # Обработка алгоритма Ахо-Корасик (из-за другой сигнатуры функции)
    if alg == "ak":
        raw = _ak_search(text_to_search, sub_list, count)
        for pattern, indices in raw.items():
            for idx in indices:
                all_matches.append((idx, pattern))
    else:
        # Обработка остальных алгоритмов через единое отображение
        alg_mapping: dict[str, Callable] = {
            'kmp': _kmp_search,
            'bm': _bm_search
        }
        func = alg_mapping[alg]
        total = 0
        limit = count if count is not None else float("inf")

        for pattern in sub_list:
            if total >= limit:
                break

            # Запрашиваем максимум оставшихся найденных индексов
            remaining = limit - total
            indices = func(text_to_search, pattern, remaining)

            for idx in indices:
                if total >= limit:
                    break

                all_matches.append((idx, pattern))
                total += 1

        all_matches.sort(key=lambda x: x[0])

    return all_matches


def _search_single_pattern_in_reversed(func: Callable,
                                       pattern_rev: str,
                                       orig_pattern: str,
                                       text_to_search: str,
                                       sub_list: List[str],
                                       limit_results: Optional[int],
                                       total: int
) -> tuple[List[tuple[int, str]], int]:
    remaining = limit_results - total if limit_results is not None else None
    indices = func(text_to_search[::-1], pattern_rev, remaining)
    matches = []
    for rev_idx in indices:
        if limit_results is not None and total >= limit_results:
            break
        orig_idx = len(text_to_search) - rev_idx - len(orig_pattern)
        matches.append((orig_idx, orig_pattern))
        total += 1
        if limit_results is not None and total >= limit_results:
            break
    return matches, total


def _process_other_last(alg: str,
                        text_to_search: str,
                        sub_list: List[str],
                        limit_results: Optional[int]
) -> List[tuple[int, str]]:
    """Обрабатывает метод 'last' с использованием других алгоритмов."""
    algo_map: dict[str, Callable] = {
        'kmp': _kmp_search,
        'bm': _bm_search,
    }
    func = algo_map[alg]
    all_matches = []
    total = 0
    rev_sub_list = [pat[::-1] for pat in sub_list]
    for pattern_rev in rev_sub_list:
        if limit_results is not None and total >= limit_results:
            break
        orig_pattern = sub_list[rev_sub_list.index(pattern_rev)]
        new_matches, total = _search_single_pattern_in_reversed(
            func, pattern_rev, orig_pattern, text_to_search, sub_list, limit_results, total
        )
        all_matches.extend(new_matches)
    all_matches.sort(key=lambda x: x[0], reverse=True)
    return all_matches

def _process_last_method(
        alg: str,
        work_string: str,
        sub_list: List[str],
        max_results: Optional[int]
) -> List[tuple[int, str]]:
    """Обрабатывает метод 'last' для поиска подстрок."""
    if alg == 'ak':
        return _process_ak_last(work_string, sub_list, max_results)
    return _process_other_last(alg, work_string, sub_list, max_results)


def _build_result(is_single: bool,
                  all_matches: List[tuple[int, str]],
                  input_substring: List[str]
) -> Optional[Union[tuple[int, ...], dict[str, tuple[int, ...]]]]:
    """
    Формирует итоговый результат.

    :param is_single (bool) True, если искали одну строку, иначе False
    :param all_matches (List[tuple[int, str]]) Cписок кортежей (позиция, нормализованная_подстрока)
    :param input_substring (List[str]) Cписок оригинальных строк, чтобы вернуть ключи в исходном виде
    :return: Кортеж индексов (для одной подстроки) или словарь
             {подстрока: кортеж индексов} (для нескольких).
    """

    # Ветка поиска по одной строке
    if is_single:
        indices = [idx for idx, _ in all_matches]
        return tuple(indices) if indices else None

    # Ветка поиска по нескольким строкам
    normal_to_original = {s.lower() if s else s: s for s in input_substring}
    matches_normal: dict[str, List[int]] = {}

    # Сопоставление нормализованной строки → с оригинальной.
    for idx, normal_pat in all_matches:
        matches_normal.setdefault(normal_pat, []).append(idx)

    # Собираем позиции по каждой нормальной подстроке.
    result_dict = {}
    for normal_pat, original in normal_to_original.items():
        matches = tuple(matches_normal.get(normal_pat, []))
        # Собираем итоговый словарь: позиции если есть, иначе None.
        result_dict[original] = matches if matches else None

    #print(result_dict)
    return None if all(v is None for v in result_dict.values()) else result_dict


@log_time_decorator
def search(string: str,
           sub_string: Union[str, list[str]],
           case_sensitivity: bool = False,
           method: str='first',
           count: Optional[int]=None,
           alg: str='kmp',
) -> Optional[Union[tuple[int, ...], dict[str, tuple[int, ...]]]]:
    """
    Ищет вхождения подстроки(ок) в строке с использованием алгоритма КМП.

    Поддерживает: чувствительность к регистру, ограничение count,
    направление поиска (first/last), поиск нескольких подстрок.

    :param string: Исходная строка.
    :param sub_string: Подстрока (str) или список/кортеж подстрок.
    :param case_sensitivity: True - учитывать регистр, False - нет.
    :param method: Направление поиска: 'first' (с начала) или 'last' (с конца).
    :param count: Максимальное количество вхождений.
    :param alg: Выбор алгоритма поиска ('kmp', 'bm', 'ak').
    :return: Кортеж индексов (для одной подстроки) или словарь
             {подстрока: кортеж индексов} (для нескольких).
    """

    # 1. Обработка чувствительности к регистру

    if alg not in SUPPORT_ALG:
        raise ValueError(f'Неподдерживаемый алгоритм {alg}')

    text_to_search = string if case_sensitivity else string.lower()
    # Проверяет, передана ли одна строка или список строк
    if isinstance(sub_string, str):
        # Ветка для одной подстроки
        work_substring = sub_string if case_sensitivity else sub_string.lower()
        is_single = True
        input_substring = [sub_string]
    else:
        # Ветка для нескольких подстрок
        work_substring = [s if case_sensitivity else s.lower() for s in sub_string]
        is_single = False
        input_substring = sub_string

    # Сформировать список подстрок для поиска в зависимости от того,
    # выполняется ли поиск одной подстроки или набора подстрок.
    if is_single:
        sub_list = [work_substring] if work_substring else []
    else:
        sub_list = [s for s in work_substring if s]

    if not sub_list:
        return None if is_single else {s: None for s in input_substring}

    if len(sub_list) > 1: alg = 'ak'

    if method == 'first':
        all_matches = _process_first_method(alg, text_to_search, sub_list,
                                            count)
    elif method == 'last':
        all_matches = _process_last_method(alg, text_to_search, sub_list,
                                            count)
    else:
        raise ValueError('Неуказан метод поиска')
    #print(f"Запрос: '{string}', {sub_string}, {case_sensitivity}, '{method}', {count}, {alg}")
    return _build_result(is_single, all_matches, input_substring)



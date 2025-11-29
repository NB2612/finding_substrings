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
from typing import Union, Optional, List, Callable, Dict
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
# ... (оставляем класс ACNode без изменений)
    __slots__ = ['children', 'fail', 'output']

    def __init__(self):
        self.children: dict[str, ACNode] = {}
        self.fail: Optional[ACNode] = None
        self.output: List[str] = []

    def __repr__(self) -> str:
        return f"AhoCorasickNode(output={self.output})"

    def add_pattern(self, pattern: str) -> None:
        node = self
        for char in pattern:
            if char not in node.children:
                node.children[char] = ACNode()
            node = node.children[char]
        node.output.append(pattern)

    def build_failure_links(self) -> None:
        queue = deque()

        for child in self.children.values():
            child.fail = self
            queue.append(child)

        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                queue.append(child)
                fail_node = current.fail
                while fail_node is not self and char not in fail_node.children:
                    fail_node = fail_node.fail
                child.fail = fail_node.children.get(char, self)
                child.output += child.fail.output

    def search_in(self, text: str, limit: Optional[int] = None) -> dict[str, List[int]]:
        """
        Выполняет поиск всех добавленных шаблонов в заданном тексте.
        Возвращает: Словарь {нормализованный_паттерн: список индексов начала вхождений}
        """
        all_patterns = set()
        stack = [self]
        visited = {self}
        while stack:
            node = stack.pop()
            if node.output:
                all_patterns.update(node.output)
            for child in node.children.values():
                if child not in visited:
                    stack.append(child)
                    visited.add(child)

        result = {pattern: [] for pattern in all_patterns}

        node = self
        total_found = 0

        for i, char in enumerate(text):
            while node is not self and char not in node.children:
                node = node.fail

            if char in node.children:
                node = node.children[char]
            else:
                node = self

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
    Возвращает: Словарь {нормализованный_паттерн: список индексов начала совпадений}
    """
    if not patterns or not text_to_search or (limit is not None and limit <= 0):
        return {p: [] for p in patterns if p}

    non_empty_patterns = [p for p in patterns if p]
    if not non_empty_patterns:
        return {}

    # Строим Trie и fail-ссылки
    root = ACNode()
    for pattern in non_empty_patterns:
        root.add_pattern(pattern)
    root.build_failure_links()

    # Выполняем поиск в тексте
    raw_result = root.search_in(text_to_search, limit)

    # raw_result уже содержит только нормализованные паттерны.
    # Добавляем пустые списки для нормализованных, но не найденных паттернов
    final_result = {p: raw_result.get(p, []) for p in patterns if p}

    return final_result


def _process_ak_last(text_to_search: str,
                     sub_list: List[str],
                     limit_res: Optional[int] = None,
                     normal_to_original_map: Dict[str, str] = None # <-- ДОБАВЛЕНО
) -> List[tuple[int, str]]:
    """
    Обрабатывает поиск с методом 'last', используя алгоритм Ахо-Корасик.
    Возвращает: Список кортежей (индекс начала, оригинальный_паттерн)
    """
    if normal_to_original_map is None:
        normal_to_original_map = {s: s for s in sub_list} # Заглушка, если нет маппинга

    # Переворачиваем паттерны и текст для поиска с конца
    rev_sub_list = [pattern[::-1] for pattern in sub_list]
    string_rev = text_to_search[::-1]

    raw = _ak_search(string_rev, rev_sub_list, limit_res)
    all_matches = []
    total = 0

    # Преобразуем индексы обратно в оригинальный текст
    # pattern_rev здесь - нормализованная и перевернутая подстрока
    for pattern_rev, indices in raw.items():
        if limit_res is not None and total >= limit_res:
            break

        # Находим нормализованный, неперевернутый паттерн
        normal_pattern = pattern_rev[::-1]
        # Находим оригинальный паттерн
        orig_pattern = normal_to_original_map.get(normal_pattern, normal_pattern)

        # Длина паттерна (должна быть нормализованная длина для расчета индекса)
        len_pattern = len(normal_pattern)

        for rev_idx in indices:
            if limit_res is not None and total >= limit_res:
                break

            # Расчет оригинального индекса
            orig_idx = len(text_to_search) - rev_idx - len_pattern

            # Добавляем кортеж с оригинальным паттерном
            all_matches.append((orig_idx, orig_pattern))
            total += 1

    # Сортируем по оригинальному индексу (0-й элемент кортежа) по убыванию
    all_matches.sort(key=lambda x: x[0], reverse=True)

    # Применяем ограничение count
    if limit_res is not None:
        all_matches = all_matches[:limit_res]

    return all_matches


# ------------------- END AHO-CORASICK -------------------

def _kmp_search(text: str, pattern: str,
                max_results: Optional[int] = None) -> List[int]:
# ... (оставляем KMP без изменений, он возвращает List[int])
    if not pattern:
        return []

    lps = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j

    results = []
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1

            if j == len(pattern):
                results.append(i - j)
                if max_results is not None and len(results) >= max_results:
                    return results

                j = lps[j - 1]

        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return results

def _bm_search(text_to_search: str,
               sub_list: str, # Это одна строка-паттерн
               limit: Optional[int] = None
) -> List[int]:
# ... (оставляем BM без изменений, он возвращает List[int])
    if not sub_list or not text_to_search:
        return []

    m = len(sub_list)
    n = len(text_to_search)
    if m > n:
        return []

    bad_char = {}
    for i, char in enumerate(sub_list):
        bad_char[char] = i

    results = []
    s = 0
    limit_val = limit if limit is not None else float('inf')

    while s <= n - m:
        j = m - 1

        while j >= 0 and sub_list[j] == text_to_search[s + j]:
            j -= 1

        if j < 0:
            results.append(s)
            if len(results) >= limit_val:
                break
            # Упрощенное правило сдвига (без хорошего суффикса)
            # Сдвиг на 1, если нет полного правила:
            s += 1
        else:
            char_shift = j - bad_char.get(text_to_search[s + j], -1)
            s += max(1, char_shift)

    return results


def _process_first_method(alg: str,
                          text_to_search: str,
                          sub_list: List[str], # Список нормализованных паттернов
                          count: Optional[int],
                          normal_to_original_map: Dict[str, str] # <-- ДОБАВЛЕНО
) -> List[tuple[int, str]]:
    """
       Выполняет поиск по тексту, возвращая первые найденные вхождения подстрок.
       Результат: List[(позиция_в_тексте, ОРИГИНАЛЬНАЯ_подстрока)]
    """
    all_matches = []

    # Обработка алгоритма Ахо-Корасик
    if alg == "ak":
        raw = _ak_search(text_to_search, sub_list, count)
        for normal_pattern, indices in raw.items():
            # Получаем оригинальную подстроку
            original_pattern = normal_to_original_map.get(normal_pattern, normal_pattern)
            for idx in indices:
                all_matches.append((idx, original_pattern)) # <-- ИСПОЛЬЗУЕМ ОРИГИНАЛ
    else:
    # Обработка остальных алгоритмов через единое отображение
        alg_mapping: dict[str, Callable] = {
            'kmp': _kmp_search,
            # Обертка для BM, чтобы она возвращала List[int]
            'bm': lambda text, pattern, max_res: _bm_search(text, pattern, max_res)
        }
        func = alg_mapping[alg]

        # 💥 ИСПРАВЛЕНИЕ: Сначала находим ВСЕ совпадения, потом сортируем и ограничиваем

        # 1. Сбор всех совпадений без ограничения count
        for normal_pattern in sub_list:
            # Получаем оригинальную подстроку
            original_pattern = normal_to_original_map.get(normal_pattern, normal_pattern)

            # Запрашиваем ВСЕ индексы (None = без ограничения)
            indices = func(text_to_search, normal_pattern, None)

            for idx in indices:
                all_matches.append((idx, original_pattern))

        # 2. Сортировка по индексу
        all_matches.sort(key=lambda x: x[0], reverse=False)

        # 3. Применение ограничения count
        if count is not None:
            all_matches = all_matches[:count]

    return all_matches


def _search_single_pattern_in_reversed(func: Callable,
                                       normal_pattern_rev: str,
                                       normal_pattern: str,
                                       text_to_search: str,
                                       limit_results: Optional[int],
                                       total: int,
                                       normal_to_original_map: Dict[str, str] # <-- ДОБАВЛЕНО
) -> tuple[List[tuple[int, str]], int]:

    remaining = limit_results - total if limit_results is not None else None
    indices = func(text_to_search[::-1], normal_pattern_rev, remaining)

    matches = []

    # Получаем оригинальную подстроку
    orig_pattern = normal_to_original_map.get(normal_pattern, normal_pattern)
    len_pattern = len(normal_pattern)

    for rev_idx in indices:
        if limit_results is not None and total >= limit_results:
            break
        orig_idx = len(text_to_search) - rev_idx - len_pattern
        matches.append((orig_idx, orig_pattern)) # <-- ИСПОЛЬЗУЕМ ОРИГИНАЛ
        total += 1
        if limit_results is not None and total >= limit_results:
            break
    return matches, total


def _process_other_last(alg: str,
                        text_to_search: str,
                        sub_list: List[str],
                        limit_results: Optional[int],
                        normal_to_original_map: Dict[str, str] # <-- ДОБАВЛЕНО
) -> List[tuple[int, str]]:
    """Обрабатывает метод 'last' с использованием KMP/BM."""
    algo_map: dict[str, Callable] = {
        'kmp': _kmp_search,
        'bm': lambda text, pattern, max_res: _bm_search(text, pattern, max_res)
    }
    func = algo_map[alg]
    all_matches = []
    total = 0

    for normal_pattern in sub_list:
        if limit_results is not None and total >= limit_results:
            break

        normal_pattern_rev = normal_pattern[::-1]

        new_matches, total = _search_single_pattern_in_reversed(
            func, normal_pattern_rev, normal_pattern, text_to_search, limit_results, total, normal_to_original_map
        )
        all_matches.extend(new_matches)

    # Сортируем по возрастанию индекса для корректного отображения
    all_matches.sort(key=lambda x: x[0], reverse=True)
    return all_matches

def _process_last_method(
        alg: str,
        work_string: str,
        sub_list: List[str],
        max_results: Optional[int],
        normal_to_original_map: Dict[str, str] # <-- ДОБАВЛЕНО
) -> List[tuple[int, str]]:
    """Обрабатывает метод 'last' для поиска подстрок."""
    if alg == 'ak':
        return _process_ak_last(work_string, sub_list, max_results, normal_to_original_map)
    return _process_other_last(alg, work_string, sub_list, max_results, normal_to_original_map)


def _build_result(is_single: bool,
                  all_matches: List[tuple[int, str]],
                  input_substring: List[str]
) -> Optional[Union[tuple[int, ...], dict[str, tuple[int, ...]]]]:
    """
    Формирует итоговый результат.
    all_matches: Cписок кортежей (позиция, ОРИГИНАЛЬНАЯ_подстрока).
    """

    # 1. Ветка поиска по одной строке (просто кортеж индексов)
    if is_single:
        # Возвращаем кортеж индексов
        indices = [idx for idx, _ in all_matches]
        return tuple(indices) if indices else None

    # 2. Ветка поиска по нескольким строкам

    # Инициализируем словарь с пустыми списками для всех оригинальных подстрок
    result_dict: dict[str, List[int]] = {s: [] for s in input_substring}

    # Группируем индексы по оригинальной подстроке
    for idx, original_pat in all_matches:
        # Паттерн в all_matches должен быть одним из input_substring
        if original_pat in result_dict:
            result_dict[original_pat].append(idx)
        # Иначе это пустая строка, которую мы отфильтровали, или что-то не так

    # Финальное преобразование: List[int] -> tuple[int, ...]
    final_result = {}
    for original_pat in input_substring:
        indices = tuple(result_dict.get(original_pat, []))

        # *** КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ДЛЯ СЛОВАРЯ:
        # В тестах ожидается None в качестве значения, если подстрока не найдена
        final_result[original_pat] = indices if indices else None
        # *** КОНЕЦ КРИТИЧЕСКОГО ИСПРАВЛЕНИЯ

    # Возвращаем словарь
    return None if all(v is None for v in final_result.values()) else final_result


@log_time_decorator
def search(string: str,
           sub_string: Union[str, list[str]],
           case_sensitivity: bool = False,
           method: str='first',
           count: Optional[int]=None,
           alg: str='kmp'
) -> Optional[Union[tuple[int, ...], dict[str, tuple[int, ...]]]]:
    """
    Ищет вхождения подстроки(ок) в строке.
    """
    if alg not in SUPPORT_ALG:
        raise ValueError(f'Неподдерживаемый алгоритм {alg}')

    # 1. Нормализация текста для поиска
    text_to_search = string if case_sensitivity else string.lower()

    input_substring: List[str]
    work_substring: Union[str, List[str]]

    # 2. Обработка подстрок
    if isinstance(sub_string, str):
        # Ветка для одной подстроки
        input_substring = [sub_string]
        is_single = True
        work_substring = sub_string if case_sensitivity else sub_string.lower()
        sub_list = [work_substring] if work_substring else []
        # Маппинг: нормализованный -> оригинальный
        normal_to_original_map = {work_substring: sub_string}
    else:
        # Ветка для нескольких подстрок
        input_substring = sub_string
        is_single = False
        # Создаем список нормализованных подстрок
        work_substring = [s if case_sensitivity else s.lower() for s in sub_string]
        # Список для поиска (без пустых строк)
        sub_list = [s for s in work_substring if s]

        # Маппинг: нормализованный -> оригинальный. Ключ должен быть взят из work_substring!
        normal_to_original_map = {}
        for i, normal_pat in enumerate(work_substring):
            # Проверяем на пустую строку, прежде чем добавить в маппинг
            if normal_pat and normal_pat not in normal_to_original_map:
                normal_to_original_map[normal_pat] = input_substring[i]


    # 3. Фильтрация и выбор алгоритма
    if not sub_list:
        return None if is_single else {s: None for s in input_substring}

    # Если подстрок много, форсируем AK
    if len(sub_list) > 1 and alg != 'ak':
         #print(f"Предупреждение: Для нескольких подстрок ({len(sub_list)})\
         # используется алгоритм Ахо-Корасик (ak) вместо {alg}.")
         alg = 'ak'

    # 4. Выполнение поиска
    if method == 'first':
        all_matches = _process_first_method(alg, text_to_search, sub_list,
                                            count, normal_to_original_map) # <-- ПЕРЕДАЕМ МАППИНГ
    elif method == 'last':
        all_matches = _process_last_method(alg, text_to_search, sub_list,
                                            count, normal_to_original_map) # <-- ПЕРЕДАЕМ МАППИНГ
    else:
        raise ValueError('Неуказан метод поиска')

    # 5. Формирование результата
    return _build_result(is_single, all_matches, input_substring)
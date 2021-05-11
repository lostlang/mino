from tests import tests
import numpy

def sort_part(rectangles: tuple, l_mino: tuple) -> dict:
    parts = dict()

    for i in rectangles:
        max_len = max(i[0][0], i[0][1])
        wight = 2 * (i[0][0] + i[0][1])
        try:
           parts[max_len][wight].append((0, *i))
        except KeyError:
            try:
                parts[max_len][wight] = [(0, *i)]
            except KeyError:
                parts[max_len] = dict()
                parts[max_len][wight] = [(0, *i)]

    for i in l_mino:
        max_len = max(i[0][0], i[0][1])
        wight = 2 * (i[0][0] + i[0][1])
        try:
            parts[max_len][wight].append((1, *i))
        except KeyError:
            try:
                parts[max_len][wight] = [(1, *i)]
            except KeyError:
                parts[max_len] = dict()
                parts[max_len][wight] = [(1, *i)]

    parts_stack = []

    i = 0
    i2 = 0
    i3 = 0
    i4 = 0
    i5 = 0
    amount = get_amount(parts)
    sort_values = sorted(parts, reverse=True)
    sort_values2 = sorted(parts[sort_values[i2]], reverse=True)

    while i < amount:
        parts_stack.append(
            parts[sort_values[i2]][sort_values2[i3]][i4][:-1]
        )
        i += 1
        i5 += 1
        if i5 == parts[sort_values[i2]][sort_values2[i3]][i4][-1]:
            i4 += 1
            i5 = 0
        if i4 == len(parts[sort_values[i2]]
                     [sort_values2[i3]]):
            i3 += 1
            i4 = 0
        if i3 == len(sort_values2):
            i2 += 1
            i3 = 0
            try:
                sort_values2 = sorted(parts[sort_values[i2]], reverse=True)
            except IndexError:
                pass
    return parts_stack


def get_amount(parts: dict) -> int:
    amount = 0
    for i in parts:
        for i2 in parts[i]:
            for part in parts[i][i2]:
                amount += part[-1]
    return amount


def draw_map(size_map: list,
             figures: list) -> numpy.array:
    mino_map = numpy.zeros([x + 2 for x in size_map], dtype=numpy.uint8)

    # to place
    mino_map[-2] = 1
    mino_map[:, -2] = 1
    mino_map[:, 1] = 1
    mino_map[1] = 1

    # border
    mino_map[-1] = 2
    mino_map[:, -1] = 2
    mino_map[:, 0] = 2
    mino_map[0] = 2

    for figure in figures:
        draw_mino(mino_map, figure)

    return mino_map


def draw_mino(mino_map: numpy.array,
              figure: tuple):
    """
    figure: object contains type, size(h, w), rotate status, flip status, point start(x, y)
    """
    part_map = numpy.zeros([x + 2 for x in figure[1]], dtype=numpy.uint8)
    # L-mino
    if figure[0] == 1:
        # L
        part_map[1:-1, 1] = 2
        part_map[-2, 1:-1] = 2
        # border
        part_map[0, 1] = 1
        part_map[-2, -1] = 1
        part_map[1:-1, 0] = 1
        part_map[1:-2, 2] = 1
        part_map[-1, 1:-1] = 1
        part_map[-3, 2:-1] = 1
    else:
        # rectangle
        part_map[1:-1, 1:-1] = 2
        # border
        part_map[0, 1:-1] = 1
        part_map[-1, 1:-1] = 1
        part_map[1:-1, 0] = 1
        part_map[1:-1, -1] = 1
        pass

    for i in range(figure[2]):
        part_map = numpy.rot90(part_map)
    if figure[3] == 1:
        part_map = numpy.flip(part_map, 0)
    mino_map[figure[-1][0] - 1: figure[-1][0] + part_map.shape[0] - 1,
             figure[-1][1] - 1: figure[-1][1] + part_map.shape[1] - 1] = numpy.maximum(
        part_map, mino_map[figure[-1][0] - 1: figure[-1][0] + part_map.shape[0] - 1,
                           figure[-1][1] - 1: figure[-1][1] + part_map.shape[1] - 1])


def part_comparator(first_obj: numpy.array,
                    second_obj: numpy.array,
                    indexes_start_first_obj: tuple) -> int:
    try:
        if (second_obj >= first_obj[indexes_start_first_obj[0] - 1:indexes_start_first_obj[0] + second_obj.shape[0] - 1,
                                    indexes_start_first_obj[1] - 1:indexes_start_first_obj[1] + second_obj.shape[1] - 1]
        ).all(-1).all(-1):
            return sum(sum(second_obj == first_obj[
                            indexes_start_first_obj[0] - 1:indexes_start_first_obj[0] + second_obj.shape[0] - 1,
                            indexes_start_first_obj[1] - 1:indexes_start_first_obj[1] + second_obj.shape[1] - 1
                            ]))
        else:
            return 0
    except ValueError:
        return 0


def get_stack_placement_state(mino_map: numpy.array,
                              mino: list) -> list:
    """
    mino: object contains type, size(h, w)
    """
    mino_positions = dict()

    part_map = numpy.ones([x + 2 for x in mino[-1]], dtype=numpy.uint8)
    # border
    part_map[-1] = 2
    part_map[:, -1] = 2
    part_map[:, 0] = 2
    part_map[0] = 2
    # L-mino
    if mino[0] == 1:
        part_map[1:-2, 2:-1] = 2

    available_position = numpy.where(mino_map == 1)
    indexes = list(zip(available_position[0], available_position[1]))

    # rotate 90
    part_map_90 = numpy.rot90(part_map)
    for index in indexes:
        # default
        wight = part_comparator(mino_map, part_map, index)
        if wight > 0:
            try:
                mino_positions[wight].append((0, 0, index))
            except KeyError:
                mino_positions[wight] = [(0, 0, index)]

        # rotate 90
        wight = part_comparator(mino_map, part_map_90, index)
        if wight > 0:
            try:
                mino_positions[wight].append((1, 0, index))
            except KeyError:
                mino_positions[wight] = [(1, 0, index)]

    # only for  L-mino #
    if mino[0] == 1:
        part_map_180 = numpy.rot90(part_map_90)
        part_map_270 = numpy.rot90(part_map_180)
        part_map_flip = numpy.flip(part_map_270, 0)
        part_map_flip_90 = numpy.rot90(part_map_flip)
        part_map_flip_180 = numpy.rot90(part_map_flip_90)
        part_map_flip_270 = numpy.rot90(part_map_flip_180)

        for index in indexes:
            # rotate 180
            wight = part_comparator(mino_map, part_map_180, index)
            if wight > 0:
                try:
                    mino_positions[wight].append((2, 0, index))
                except KeyError:
                    mino_positions[wight] = [(2, 0, index)]
            # rotate 270
            wight = part_comparator(mino_map, part_map_270, index)
            if wight > 0:
                try:
                    mino_positions[wight].append((3, 0, index))
                except KeyError:
                    mino_positions[wight] = [(3, 0, index)]
            # flip
            wight = part_comparator(mino_map, part_map_flip, index)
            if wight > 0:
                try:
                    mino_positions[wight].append((3, 1, index))
                except KeyError:
                    mino_positions[wight] = [(3, 1, index)]
            # rotate 90
            wight = part_comparator(mino_map, part_map_flip_90, index)
            if wight > 0:
                try:
                    mino_positions[wight].append((2, 1, index))
                except KeyError:
                    mino_positions[wight] = [(2, 1, index)]
            # rotate 180
            wight = part_comparator(mino_map, part_map_flip_180, index)
            if wight > 0:
                try:
                    mino_positions[wight].append((1, 1, index))
                except KeyError:
                    mino_positions[wight] = [(1, 1, index)]
            # rotate 270
            wight = part_comparator(mino_map, part_map_flip_270, index)
            if wight > 0:
                try:
                    mino_positions[wight].append((0, 1, index))
                except KeyError:
                    mino_positions[wight] = [(0, 1, index)]

    sort_values = sorted(mino_positions)

    # drop useless
    useless_c = sum(mino[-1]) * 2
    try:
        while sort_values[-1] < useless_c:
            sort_values.pop()
    except IndexError:
        return []

    stack_positions = []
    for value in sort_values:
        for position in mino_positions[value]:
            stack_positions.append(position)
    return stack_positions


def list_with_index_to_list(input: list) -> list:
    output = []
    for i in input:
        output.append(i[0][i[1]])
    return output


def get_cute_info(*data):
    text = f"Test Data: \n" \
           f"\tSize rectangle {data[0][0]} \n" \
           f"\tR(Rectangle)-mino : {data[0][1]}\n" \
           f"\tL-mino : {data[0][2]}\n" \
           f"Answer : {data[1]}; Right answer : {data[0][-1]}.\n" \
           f"Iterations : {data[2]} \n"
    if data[1]:
        text += f"Path : \n"
    name_mino = ["R", "L"]
    for i in data[-1]:
        tmp = f"\t {name_mino[i[0]]}-mino {i[1]} "
        if i[2] > 0:
            tmp += f"rotate {90*i[2]} grad. "
        if i[3] == 1:
            tmp += f"flip horizontal, "
        tmp += f"move to {i[4]} \n"
        text += tmp
    else:
        text = text[:-1]
    return text


def part_size(parts: list) -> int:
    size = 0
    for part in parts:
        if part[0] == 1:
            size += part[-1][0] + part[-1][1] - 1
        else:
            size += part[-1][0] * part[-1][1]
    return size


def search_placing(test_data: list,
                   cute_data=None):
    possibly = False
    i = 0
    k = 0
    cute_flag = True
    if cute_data is not None:
        if not cute_data:
            cute_flag = False

    parts = sort_part(test_data[1], test_data[2])
    if part_size(parts) > test_data[0][0] * test_data[0][1]:
        if cute_flag:
            return get_cute_info(test_data, possibly, k, [])
        else:
            return possibly
    mino_map = draw_map(test_data[0], [])

    mino_placement = get_stack_placement_state(mino_map, parts[0])
    list_mino = []

    while i < len(parts):
        new_placement = get_stack_placement_state(mino_map, parts[i])

        # drop bad value

        i2 = i - 1
        while list_mino and i2 > -1 and parts[i2] == parts[i]:
            if list_mino[i2][1] + 1 != len(list_mino[i2][0]):
                drop_values = list_mino[i2][0][list_mino[i2][-1]:]
            else:
                i2 -= 1
                continue
            for bad_value in drop_values:
                if bad_value in new_placement:
                    del new_placement[new_placement.index(bad_value)]
            i2 -= 1

        if new_placement:
            if not list_mino:
                new_placement.reverse()
                new_placement = new_placement[:min(2, len(new_placement))]
                list_mino.append([new_placement, len(new_placement) - 1])
            else:
                list_mino.append([new_placement, len(new_placement) - 1])
        else:
            list_mino[-1][1] -= 1
            while list_mino[-1][1] == -1:
                list_mino.pop()
                i -= 1
                try:
                    list_mino[-1][1] -= 1
                except IndexError:
                    break
            if len(list_mino) == 0:
                break
            mino_map = draw_map(test_data[0], [(*i[0], *i[1]) for i in zip(parts, list_with_index_to_list(list_mino))])
            continue
        mino = [(*i[0], *i[1]) for i in zip(parts, list_with_index_to_list(list_mino))]
        draw_mino(mino_map, (mino[-1]))
        i += 1
        k += 1
    else:
        possibly = True

    if cute_flag:
        return get_cute_info(test_data, possibly, k,
                         [(*i[0], *i[1]) for i in zip(parts, list_with_index_to_list(list_mino))])
    return possibly


if __name__ == "__main__":
    for test in tests:
        print("Start Test")
        print(search_placing(test))
        print("End Test \n")


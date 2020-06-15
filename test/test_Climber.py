
from genealogy_aligner import Pedigree, Climber


def test_left_climber_single_source():
    ped = Pedigree.from_balsac_table('data/test/simple.tsv')

    path_taken = [8]
    climber = Climber(ped, [8])
    for node, parents in climber:
            # chose left parent
            chosen_parent = min(parents)
            climber.queue(chosen_parent)
            path_taken.append(chosen_parent)

    assert path_taken == [8, 5, 1]

    

def test_right_climber():
    ped = Pedigree.from_balsac_table('data/test/simple.tsv')

    path_taken = [7, 8]
    climber = Climber(ped, [7, 8])
    for node, parents in climber:
        if parents:
            # chose right parent
            chosen_parent = max(parents)
            climber.queue(chosen_parent)
            path_taken.append(chosen_parent)

    assert path_taken == [7, 8, 4, 6, 2]

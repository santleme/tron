"""Final-only opponent binding; omitted from all nonfinal Kaggle packages."""


def final_opponent_policy():
    from training.eval.lc_eval import open_space_action

    return open_space_action

import glob
import os


def cleanup_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        files = glob.glob(os.path.join(path, '*'))
        for f in files:
            try:
                os.remove(f)
            except:
                pass

def print_info(buffer, agents_learning_cycle, meta_agent_learning_cycle):
    scores, modified_score, modified_score2, constraints = buffer.mean_score()
    print('Meta-Agent lc {}\t Agents lc {}\t Score: {:.2f}, '
          'Modified Score: {:.2f}, {:.2f}, Constraints: {}'.format(meta_agent_learning_cycle,
                                                           agents_learning_cycle, scores,
                                                           modified_score,modified_score2, constraints))
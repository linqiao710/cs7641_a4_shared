from envs.frozen_lake import FrozenLakeEnv
import mdptoolbox.example

problem_to_process = [
    {
        'name': 'Policy_Iteration',
        'path': 'pi',
        'process': True
    },
    {
        'name': 'Value_Iteration',
        'path': 'vi',
        'process': True
    },
    {
        'name': 'Q_Learning',
        'path': 'ql',
        'process': False
    }
]

envs_to_process = [
    {
        'name': 'Frozen_Lake: 4x4',
        'path': 'fl',
        'size': '4x4',
        'type': 'grid',
        'instance': FrozenLakeEnv(map_name='4x4'),
        'process': False
    },
    {
        'name': 'Frozen_Lake: 25x25',
        'path': 'fl',
        'size': '25x25',
        'type': 'grid',
        'instance': FrozenLakeEnv(map_name='25x25'),
        'process': False
    },
    {
        'name': 'Forest_Management: S=6',
        'path': 'fm',
        'size': 's=6',
        'type': 'non-grid',
        'instance': mdptoolbox.example.forest(S=6, r1=10, r2=50),
        'process': False
    },
    {
        'name': 'Forest_Management: S=500',
        'path': 'fm',
        'size': 's=500',
        'type': 'non-grid',
        'instance': mdptoolbox.example.forest(S=500, r1=10, r2=50),
        'process': True
    }
]

random_state = 10


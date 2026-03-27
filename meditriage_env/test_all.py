import requests

difficulties = ['easy', 'medium', 'hard']

for difficulty in difficulties:
    reset = requests.post('http://127.0.0.1:8000/reset?difficulty=' + difficulty).json()
    print('\n[' + difficulty.upper() + ']')
    print('Symptoms: ' + str(reset['symptoms'][:2]))
    print('Resources: ' + str(reset['hospital_resources']))
    
    step = requests.post('http://127.0.0.1:8000/step', json={
        'priority_level': 1,
        'allocation': 'icu',
        'reasoning': 'Patient shows critical symptoms'
    }).json()
    print('Reward: ' + str(step['reward']))
    print('Condition: ' + str(step['info']['actual_condition']))
    print('Feedback: ' + str(step['info']['feedback']))
import sys
import os
import random
import pickle
sys.path.append('./model/')
sys.path.append('./')

import yaml
import torch
import numpy as np

from model.plain_transformer import PlainTransformer
from convert2midi import skyline_event_to_midi, TempoEvent
from utils import pickle_load
from inference_utils import generate_plain_xl, user_input

def get_valid_tempos(event2idx):
    """Extract all valid tempo values from the event2idx dictionary."""
    return [int(event.split('_')[1]) for event in event2idx if event.startswith("Tempo_")]

def closest_tempo(user_tempo, valid_tempos):
    """Find the closest valid tempo to the user's input."""
    return min(valid_tempos, key=lambda x: abs(x - user_tempo))

def read_vocab(vocab_file):
    event2idx, idx2event = pickle_load(vocab_file)
    orig_vocab_size = len(event2idx)
    pad_token = orig_vocab_size
    event2idx['PAD_None'] = pad_token
    vocab_size = pad_token + 1

    return event2idx, idx2event, vocab_size

def dump_midi(words, idx2event, output_midi_path=None, rfreq_cls=None, polyph_cls=None, output_event_path=None, return_tempo=False, enforce_tempo_val=None):
    events = [idx2event[w] for w in words]

    if output_event_path is not None:
        with open(output_event_path, 'w') as f:
            if rfreq_cls is not None:
                f.write('[rhymfreq] ')
                f.write(str(rfreq_cls))
                f.write('\n')
            if polyph_cls is not None:
                f.write('[polyph  ] ')
                f.write(str(polyph_cls))
                f.write('\n')
            f.write('======================================================================\n')
            print(*events, sep='\n', file=f)

    if return_tempo:
        return skyline_event_to_midi(events, output_midi_path=output_midi_path, return_tempo=True)[1]
    elif enforce_tempo_val is not None:
        skyline_event_to_midi(events, output_midi_path=output_midi_path, enforce_tempo=True, enforce_tempo_val=enforce_tempo_val)
    else:
        skyline_event_to_midi(events, output_midi_path=output_midi_path)

def get_leadsheet_prompt(data_dir, piece, prompt_n_bars):
    bar_pos, evs = pickle_load(os.path.join(data_dir, piece + '.pkl'))

    prompt_evs = [f"{x['name']}_{x['value']}" for x in evs[:bar_pos[prompt_n_bars] + 1]]
    assert len(np.where(np.array(prompt_evs) == 'Bar_None')[0]) == prompt_n_bars + 1
    target_bars = len(bar_pos)

    return prompt_evs, target_bars

if __name__ == '__main__':
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    event2idx, idx2event, vocab_size = read_vocab(config['data']['vocab_path'])

    # Extract valid tempos from the vocabulary
    valid_tempos = get_valid_tempos(event2idx)

    max_bars = user_input('max_bars')
    temp = user_input('temp')
    top_p = 0.97
    max_dec_len = 2400
    print('[nucleus parameters] t = {}, p = {}'.format(temp, top_p))

    torch.cuda.device(config['device'])

    # for generation w/ melody prompts
    use_prompt = False
    prompt_bars = 8

    mconf = config['model']
    model = PlainTransformer(
        mconf['d_word_embed'],
        vocab_size,
        mconf['decoder']['n_layer'],
        mconf['decoder']['n_head'],
        mconf['decoder']['d_model'],
        mconf['decoder']['d_ff'],
        mconf['decoder']['tgt_len'],
        mconf['decoder']['tgt_len'],
        dec_dropout=mconf['decoder']['dropout'],
        pre_lnorm=mconf['pre_lnorm']
    ).cuda()
    print('[info] # params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    pretrained_dict = torch.load(config['inference_param_path'], map_location='cpu')
    model.load_state_dict(pretrained_dict)
    model.eval()

    generated_pieces = 0
    total_pieces = n_pieces
    gen_times = []

    while generated_pieces < n_pieces:
        piece_id = generated_pieces + 1

        out_name = f'samp_{piece_id:02d}'
        if os.path.exists(os.path.join(out_dir, out_name + '.mid')):
            print(f'[info] {out_name} exists, skipping ...')
            continue

        user_tempo = user_input('tempo')  # This is the original user input for tempo
        closest_valid_tempo = closest_tempo(user_tempo, valid_tempos)  # Adjusting to the closest valid tempo

        orig_tempos = [TempoEvent(closest_valid_tempo, 0, 0)]
        print('[global tempo]', orig_tempos[0].tempo)

        print(f' -- generating leadsheet #{generated_pieces + 1} of {total_pieces}')

        gen_words, t_sec = generate_plain_xl(
            model,
            event2idx, idx2event,
            max_events=max_dec_len, max_bars=max_bars,
            primer=['Tempo_{}'.format(orig_tempos[0].tempo), 'Bar_None'],
            temp=temp, top_p=top_p
        )

        if gen_words is None:  # model failed repeatedly
            continue
        if len(gen_words) >= max_dec_len:
            continue
        if len(np.where(np.array(gen_words) == event2idx['Bar_None'])[0]) >= max_bars:
            continue

        dump_midi(
            gen_words, idx2event,
            os.path.join(out_dir, out_name + '.mid'),
            output_event_path=os.path.join(out_dir, out_name + '.txt'),
            enforce_tempo_val=orig_tempos
        )

        gen_times.append(t_sec)
        generated_pieces += 1

    print(f'[info] finished generating {generated_pieces} pieces, avg. time: {np.mean(gen_times):.2f} +/- {np.std(gen_times):.2f} secs.')

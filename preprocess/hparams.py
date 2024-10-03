import argparse
from text.symbols import symbols

def create_hparams():
    """Create model hyperparameters."""
    parser = argparse.ArgumentParser()

    ################################
    # Experiment Parameters        #
    ################################
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--iters_per_checkpoint', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dynamic_loss_scaling', type=bool, default=True)
    parser.add_argument('--fp16_run', type=bool, default=False)
    parser.add_argument('--distributed_run', type=bool, default=False)
    parser.add_argument('--dist_backend', type=str, default="nccl")
    parser.add_argument('--dist_url', type=str, default="tcp://localhost:54321")
    parser.add_argument('--cudnn_enabled', type=bool, default=True)
    parser.add_argument('--cudnn_benchmark', type=bool, default=False)
    parser.add_argument('--ignore_layers', type=list, default=['speaker_embedding.weight'])

    ################################
    # Data Parameters             #
    ################################
    parser.add_argument('--training_files', type=str, default='filelists/ljs_audiopaths_text_sid_train_filelist.txt')
    parser.add_argument('--validation_files', type=str, default='filelists/ljs_audiopaths_text_sid_val_filelist.txt')
    parser.add_argument('--text_cleaners', type=list, default=['english_cleaners'])
    parser.add_argument('--p_arpabet', type=float, default=1.0)
    parser.add_argument('--cmudict_path', type=str, default="E:\\FYp R And D\\implemetation - Copy\\preprocess\\data\\cmu_dictionary\\cmudict-0 (4).7b")

    ################################
    # Audio Parameters             #
    ################################
    parser.add_argument('--max_wav_value', type=float, default=32768.0)
    parser.add_argument('--sampling_rate', type=int, default=22050)
    parser.add_argument('--filter_length', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--win_length', type=int, default=1024)
    parser.add_argument('--n_mel_channels', type=int, default=80)
    parser.add_argument('--mel_fmin', type=float, default=0.0)
    parser.add_argument('--mel_fmax', type=float, default=8000.0)
    parser.add_argument('--f0_min', type=float, default=80)
    parser.add_argument('--f0_max', type=float, default=880)
    parser.add_argument('--harm_thresh', type=float, default=0.25)

    ################################
    # Model Parameters             #
    ################################
    parser.add_argument('--n_symbols', type=int, default=len(symbols))
    parser.add_argument('--symbols_embedding_dim', type=int, default=512)

    # Encoder parameters
    parser.add_argument('--encoder_kernel_size', type=int, default=5)
    parser.add_argument('--encoder_n_convolutions', type=int, default=3)
    parser.add_argument('--encoder_embedding_dim', type=int, default=512)

    # Decoder parameters
    parser.add_argument('--n_frames_per_step', type=int, default=1)  # currently only 1 is supported
    parser.add_argument('--decoder_rnn_dim', type=int, default=1024)
    parser.add_argument('--prenet_dim', type=int, default=256)
    parser.add_argument('--prenet_f0_n_layers', type=int, default=1)
    parser.add_argument('--prenet_f0_dim', type=int, default=1)
    parser.add_argument('--prenet_f0_kernel_size', type=int, default=1)
    parser.add_argument('--prenet_rms_dim', type=int, default=0)
    parser.add_argument('--prenet_rms_kernel_size', type=int, default=1)
    parser.add_argument('--max_decoder_steps', type=int, default=1000)
    parser.add_argument('--gate_threshold', type=float, default=0.5)
    parser.add_argument('--p_attention_dropout', type=float, default=0.1)
    parser.add_argument('--p_decoder_dropout', type=float, default=0.1)
    parser.add_argument('--p_teacher_forcing', type=float, default=1.0)

    # Attention parameters
    parser.add_argument('--attention_rnn_dim', type=int, default=1024)
    parser.add_argument('--attention_dim', type=int, default=128)

    # Location Layer parameters
    parser.add_argument('--attention_location_n_filters', type=int, default=32)
    parser.add_argument('--attention_location_kernel_size', type=int, default=31)

    # Mel-post processing network parameters
    parser.add_argument('--postnet_embedding_dim', type=int, default=512)
    parser.add_argument('--postnet_kernel_size', type=int, default=5)
    parser.add_argument('--postnet_n_convolutions', type=int, default=5)

    # Speaker embedding
    parser.add_argument('--n_speakers', type=int, default=123)
    parser.add_argument('--speaker_embedding_dim', type=int, default=128)

    # Reference encoder
    parser.add_argument('--with_gst', type=bool, default=True)
    parser.add_argument('--ref_enc_filters', type=list, default=[32, 32, 64, 64, 128, 128])
    parser.add_argument('--ref_enc_size', type=list, default=[3, 3])
    parser.add_argument('--ref_enc_strides', type=list, default=[2, 2])
    parser.add_argument('--ref_enc_pad', type=list, default=[1, 1])
    parser.add_argument('--ref_enc_gru_size', type=int, default=128)

    # Style Token Layer
    parser.add_argument('--token_embedding_size', type=int, default=256)
    parser.add_argument('--token_num', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=8)

    ################################
    # Optimization Hyperparameters #
    ################################
    parser.add_argument('--use_saved_learning_rate', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_min', type=float, default=1e-5)
    parser.add_argument('--learning_rate_anneal', type=int, default=50000)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip_thresh', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mask_padding', type=bool, default=True)  # set model's padded outputs to padded values

    args = parser.parse_args([])  # Pass an empty list to parse_args to avoid errors

    return args
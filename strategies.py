strategies = {

    'HMPCompCorScrub': {'strategy': ['high_pass', 'motion', 'compcor', 'scrub'], 
                         'motion': 'basic', 
                         'compcor': 'anat', 
                         'n_compcor': 'auto', 
                         'acompcor_combined': True,
                         'scrub': 'basic', 
                         'fd_thresh': 0.9, 
                         'std_dvars_thresh': 3.0},

    'HMPCompCorScrubGS': {'strategy': ['high_pass', 'motion', 'compcor', 'scrub', 'global'], 
                         'motion': 'basic', 
                         'compcor': 'anat', 
                         'n_compcor': 'auto', 
                         'acompcor_combined': True, 
                         'scrub': 'basic',
                         'fd_thresh': 0.9, 
                         'std_dvars_thresh': 3.0, 
                         'global_signal': 'basic'},               

    'HMPWMCSFScrub': {'strategy': ['high_pass', 'motion', 'wm_csf', 'scrub'], 
                         'motion': 'basic', 
                         'wm_csf': 'full',
                         'scrub': 'basic',
                         'fd_thresh': 0.9, 
                         'std_dvars_thresh': 3.0},

    'HMPWMCSFScrubGS': {'strategy': ['high_pass', 'motion', 'wm_csf', 'scrub', 'global'], 
                         'motion': 'basic', 
                         'wm_csf': 'full',
                         'scrub': 'basic',
                         'fd_thresh': 0.9, 
                         'std_dvars_thresh': 3.0,
                         'global_signal': 'basic'}
    }

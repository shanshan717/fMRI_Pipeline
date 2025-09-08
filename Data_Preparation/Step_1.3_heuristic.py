def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    return template, outtype, annotation_classes

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where"""
    info = {}

    # 不要在模板里写 "ses-" 前缀
    t1w = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_T1w')

    info[t1w] = []

    for s in seqinfo:
        desc = s.series_description.lower()
        if "t1" in desc or "mprage" in desc:
            info[t1w].append(s.series_id)

    return info
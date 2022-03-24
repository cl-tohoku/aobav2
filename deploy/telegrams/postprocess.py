from argparse import Namespace
from typing import List, Any, Dict

from logzero import logger

from postprocess_scorer import PostprocessScorer


def re_ranking_result(full_contexts: List[str], contexts: List[str], response: str, next_utterance_score: float):
    """This is the function for checking the content of an element"""
    return {
        "full_contexts": full_contexts,
        "contexts": contexts,
        "response": response,
        "next_utterance_score": next_utterance_score,
    }


def median(lst):
    n = len(lst)
    return sorted(lst)[n // 2] if n % 2 == 1 else sum(sorted(lst)[n // 2 - 1:n // 2 + 1]) / 2.0


def filter_by_score(
        re_ranking_results: List[Dict[str, Any]],
        thr: float,
        keyname: str,
        eliminate_lower: bool = True,
        incremental_threshold: bool = False,
        incremental_threshold_value: float = 0.05,
):
    """
    Filter out cands with lower (higher) score of 'keyname'
    """
    score_lis = [result[keyname] for result in re_ranking_results]
    if not incremental_threshold:
        # threshold is updated to median value
        score_median = median(score_lis)
        if (eliminate_lower and score_median < thr) or ((not eliminate_lower) and score_median > thr):
            logger.info('Too high/low {} thr! Temporarily lower this thr.'.format(keyname))
            thr = score_median

    accept_lis, filt_out_lis = [], []
    for result in re_ranking_results:
        if (eliminate_lower and result[keyname] >= thr) or ((not eliminate_lower) and result[keyname] <= thr):
            accept_lis.append(result)
        else:
            filt_out_lis.append(result)

    if incremental_threshold and not accept_lis:
        logger.info(f"Threshold '{thr}' is updated to '{thr + incremental_threshold_value}'")
        thr += incremental_threshold_value
        accept_lis, filt_out_lis = filter_by_score(
            re_ranking_results=re_ranking_results,
            thr=thr,
            keyname=keyname,
            eliminate_lower=eliminate_lower,
            incremental_threshold=incremental_threshold,
            incremental_threshold_value=incremental_threshold_value
        )

    return accept_lis, filt_out_lis


def display_postprocess_log(re_ranking_results, main_msg: str):
    if not re_ranking_results:
        return
    log_message = main_msg + '\n'
    log_message += '\n'.join(
        ", ".join([
            f"LM_score:{result['next_utterance_score']:.3f}",
            f"Info_score:{result['inf']:.3f}",
            #f"Sim_score:{result['highest_jaccard']:.3f}",
            f"Sim_usr_score:{result['highest_jaccard_usr']:.3f}",
            #f"Sim_mix_score:{result['highest_jaccard_mix']:.3f}",
            f"duplicate:{result['duplicate_score']:.3f}",
            f"jnsli:{result['jsnli_label']: >13}",
            f"prev_jnsli:{result['prev_jsnli_label']: >13}",
            result["response"]
        ])
        for result in re_ranking_results
    )
    logger.info(log_message)


def post_processing(
        re_ranking_results: List[Dict[str, Any]],
        postprocess_scorer: PostprocessScorer,
        args: Namespace,
) -> str:
    """
    Args:
        re_ranking_results:
            This is the list of 're_ranking_result' (= dictionary).
            're_ranking_result' contains the following elements.
                - contexts: List of str
                - response: str
                - next_utterance_score: float
        postprocess_scorer:
            This is the class for scoring.
        args:
            argparse args
    Return:
        response
    """
    accepted_results = postprocess_scorer(re_ranking_results)
    jsnli_filt_out_results,prev_jsnli_filt_out_results = [],[]

    # add filtering by jsnli label (2021/10/05 kishinami)
    # accepted_results = sorted(accepted_results, key=lambda x: x['jsnli_score'], reverse=True)
    if args.filter_by_jsnli:
        jsnli_filt_out_results = [result for result in accepted_results if result['jsnli_label'] == 'contradiction']
        accepted_results = [result for result in accepted_results if result['jsnli_label'] != 'contradiction']
        if not accepted_results:
            accepted_results = jsnli_filt_out_results
            jsnli_filt_out_results = []

    if args.filter_by_prev_jsnli:
        prev_jsnli_filt_out_results = [result for result in accepted_results if result['prev_jsnli_label'] == 'contradiction']
        accepted_results = [result for result in accepted_results if result['prev_jsnli_label'] != 'contradiction']
        if not accepted_results:
            accepted_results = prev_jsnli_filt_out_results
            prev_jsnli_filt_out_results = []
    # add filtering by jsnli label (2021/10/05 kishinami)

    # Filter out candidates according to postprocess scores
    accepted_results, next_uttr_filt_out_results = filter_by_score(
        re_ranking_results=accepted_results,
        thr=args.next_uttr_thr,
        keyname='next_utterance_score',
    )
    accepted_results, duplicate_score_out_results = filter_by_score(
        re_ranking_results=accepted_results,
        thr=args.duplicate_score_thr,
        keyname='duplicate_score'
    )
    accepted_results, jac_mix_filt_out_results = filter_by_score(
        re_ranking_results=accepted_results,
        thr=args.mix_jac_thr,
        keyname='highest_jaccard_mix',
        eliminate_lower=False,
        incremental_threshold=True,
        incremental_threshold_value=args.incremental_thr_value,
    )
    # accepted_results, inf_filt_out_results = filter_by_score(
    #     re_ranking_results=accepted_results,
    #     thr=args.inf_thr,
    #     keyname='inf'
    # )
    # accepted_results, jac_filt_out_results = filter_by_score(
    #     re_ranking_results=accepted_results,
    #     thr=args.jac_thr,
    #     keyname='highest_jaccard',
    #     eliminate_lower=False
    # )
    # accepted_results, jac_usr_filt_out_results = filter_by_score(
    #     re_ranking_results=accepted_results,
    #     thr=args.user_jac_thr,
    #     keyname='highest_jaccard_usr',
    #     eliminate_lower=False
    # )

    display_postprocess_log(accepted_results, 'accepted response:')
    display_postprocess_log(jsnli_filt_out_results, 'eliminated due to jsnli label(contradiction):')
    display_postprocess_log(prev_jsnli_filt_out_results, 'eliminated due to prev jsnli label(contradiction):')
    display_postprocess_log(next_uttr_filt_out_results, 'eliminated due to next utterance score:')
    display_postprocess_log(duplicate_score_out_results, 'eliminated due to word duplicate score:')
    display_postprocess_log(jac_mix_filt_out_results, 'eliminated due to full contexts jaccard score:')
    # display_postprocess_log(inf_filt_out_results, 'eliminated due to info score:')
    # display_postprocess_log(jac_filt_out_results, 'eliminated due to jaccard score:')
    # display_postprocess_log(jac_usr_filt_out_results, 'eliminated due to userside jaccard score:')

    # Choose response from accepted candidates
    if args.select_by_information:
        sorted_responses = sorted(accepted_results, key=lambda x: x['inf'], reverse=True)
        response = sorted_responses[0]['response']
        logger.info("The response was chosen to have the highest 'inf' from accepted responses.")
    else:
        response = accepted_results[0]["response"]
        logger.info("The response was chosen to have the highest 'next_utterance_score' from accepted responses.")

    return response

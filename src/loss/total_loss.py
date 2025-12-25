def total_loss(ic_preds, final_pred, target, tau=None):
    from ic_loss import ic_loss
    from final_loss import final_loss
    
    if tau is None:
        tau = [1.0] * len(ic_preds)
    
    loss = sum(t * ic_loss(ic, target) for t, ic in zip(tau, ic_preds))
    loss += final_loss(final_pred, target)
    
    return loss

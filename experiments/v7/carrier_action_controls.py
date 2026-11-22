"""Check first-action information against equally map-informed baselines."""
import carrier_action_fit as fit
import carrier_action as c
import argparse,gzip,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,default=fit.ROOT/'analysis/v7/carrier-action-controls.json');a=p.parse_args()
    model=dict(np.load(fit.HERE/'carrier-action-model.npz'))
    factual=json.loads((fit.ROOT/'analysis/v7/carrier-action-factual.json').read_text())
    lookup={r['layout']:Path(r['source'])/'replays'/r['file'] for r in json.loads(a.manifest.read_text())['games']}
    split={}
    for fold in ('train','test'):
        features=[];labels=[];actual=[];blind=[]
        for r in factual[fold+'_maps']:
            cache=a.cache/(r['canonical']+'-allmoves.npz');y=np.load(a.cache/(r['canonical']+'-current.npz'))['y']
            if cache.exists():
                with np.load(cache) as f:xx=f['rows'];actions=f['actions']
            else:
                with gzip.open(lookup[r['layout']]) as f:g=json.load(f)
                frames=g['steps'];rows=[];cmd=[]
                for t in range(5,len(frames)-1,4):
                    pl=frames[t][0]['observation']['players']
                    for side in (0,1):
                        ob={'player':side,'players':pl,'step':t,'carbon':frames[t][side]['observation']['carbon']}
                        ctx=c._ca_board(ob);action=frames[t+1][side]['action'] or {}
                        for uid,(pos,load,kind) in pl[side][2].items():
                            if kind!='COLLECTOR' or load<60:continue
                            rows.append(c._ca_inputs(ctx,uid,range(5))[0]);cmd.append(c.NAMES.index(action.get(uid) or 'NONE'))
                xx=np.asarray(rows,'f');actions=np.asarray(cmd,np.uint8);np.savez_compressed(cache,rows=xx,actions=actions)
            assert len(xx)==len(y)
            # This state-only model knows the same five possible landing feature vectors but is never told which was selected.
            feat=np.concatenate([xx[:,0,:],xx[:,:,16:].min(1),xx[:,:,16:].max(1),xx[:,:,16:].mean(1)],axis=1)
            features.append(feat);labels.append(y)
            if fold=='test':
                preds=np.asarray([c._ca_eval(model,xx.reshape(-1,xx.shape[-1]),h).reshape(len(xx),5) for h in ('10','20')]).transpose(1,2,0)
                preds[:,:,1]=np.maximum(preds[:,:,0],preds[:,:,1])
                actual.append(preds[np.arange(len(xx)),actions]);blind.append(preds.mean(axis=1))
        split[fold]=np.concatenate(features),np.concatenate(labels)
        if fold=='test':fact=np.concatenate(actual);uniform=np.concatenate(blind)
    train,y=split['train'];test,z=split['test'];est=[]
    for i in range(2):
        reg=HistGradientBoostingRegressor(max_iter=66,max_depth=4,learning_rate=.10,max_leaf_nodes=14,min_samples_leaf=36,l2_regularization=16,early_stopping=False,random_state=831)
        reg.fit(train,y[:,i]);est.append(reg.predict(test))
    allmoves=np.maximum(0,np.asarray(est).T);allmoves[:,1]=np.maximum(allmoves[:,1],allmoves[:,0])
    summary={n:fit.metrics(z,value) for n,value in [('all_five_landings_but_selected_action_hidden',allmoves),('conditional_predictions_averaged_over_five',uniform),('conditional_actual_first_action',fact)]}
    assert np.allclose(summary['conditional_actual_first_action']['mae'],factual['comparisons']['all']['action_conditional']['mae'])
    a.out.write_text(json.dumps({'model_sha':factual['model_sha'],'test_maps':len(factual['test_maps']),'train_maps':len(factual['train_maps']),'metrics':summary},indent=1)+'\n')
    print(summary,flush=True)

if __name__=='__main__':main()

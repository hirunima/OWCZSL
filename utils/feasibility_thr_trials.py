   feasibility_threshold_val=evaluator_val_ge.thresholding(valset.val_seen_mask,unseen_scores)
    # feasibility_threshold_train=evaluator_val_ge.thresholding(trainset.val_seen_mask,unseen_scores)
    
    feasibility_threshold=feasibility_threshold_val
    print(f'feasibility threshold is set as: {feasibility_threshold}')
    
    checking=[[],[]]
    for i,pairesall in enumerate(valset.pairs):
        if (unseen_scores[i]<feasibility_threshold and unseen_scores[i]!=0):
           checking[1].append(pairesall) 
        else:
            checking[0].append(pairesall) 

    man_ckec=[]
    problematic =[('ancient', 'apple'),('ancient', 'balloon'), ('ancient', 'banana'),('ancient', 'beef')]
    # for i,pairesall in enumerate(valset.pairs):
    #     if pairesall in problematic:
    #         man_ckec.append((pairesall,unseen_scores[i]) )

    for i,pairesall in enumerate(valset.pairs):
        if (0<unseen_scores[i]<0.25 and valset.val_seen_mask[i]==1):
            man_ckec.append((pairesall,unseen_scores[i]))
    import pdb; pdb.set_trace()
    print(len(checking[0]),len(checking[1]))
    print('*****************************************************************')

    # feasibility_threshold=feasibility_threshold_val
    print(f'feasibility threshold is set as: {feasibility_threshold}')
    
    #just checking stufff***********************************
    checking=[[],[]]
    for i,pairesall in enumerate(valset.pairs):
        if (unseen_scores[i]<feasibility_threshold and unseen_scores[i]!=0):
           checking[1].append(pairesall) 
        else:
            checking[0].append(pairesall) 

    man_ckec=[]
    problematic =[('ancient', 'apple'),('ancient', 'balloon'), ('ancient', 'banana'),('ancient', 'beef')]
    # for i,pairesall in enumerate(valset.pairs):
    #     if pairesall in problematic:
    #         man_ckec.append((pairesall,unseen_scores[i]) )

    for i,pairesall in enumerate(testset.pairs):
        if (0<unseen_scores[i]<feasibility_threshold and testset.test_seen_mask[i]==1):
            man_ckec.append((pairesall,unseen_scores[i]))
    print(len(checking[0]),len(checking[1]))
    import pdb; pdb.set_trace()
    print('*****************************************************************')

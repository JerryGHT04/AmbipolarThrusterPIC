#index of all alive
aliveIdx = findall(x->x==1, myParticle.alive)

#index of all neutrals
neutralIdx = findall(x->x==0.0, myParticle.qArray[aliveIdx])

#k index for all particle
k = (floor.(myParticle.rCellNum).-1).*myDomain.Nz + floor.(myParticle.zCellNum)

#index of cells with neutral change
idx = findall(x->x<0,myDomain.neutral_change)

#k index for all cells with neutral change
kWithNeutralChange = (getindex.(idx,1).-1).*myDomain.Nz .+ getindex.(idx,2)

#only loop neutrals in the k cell with neutral change
vec1 = Array(kWithNeutralChange)
vec2 = Array(k)
idxkNeeded = findall(x->x in vec1, vec2)



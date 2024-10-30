import numpy as np

class DIIS():
    def __init__( self, unrestricted=False, N_DIIS=100 ):
        if ( unrestricted == True ):
            self.fock_list_a = []
            self.fock_list_b = []
            self.error_list_a = []
            self.error_list_b = []
        else:
            self.fock_list = []
            self.error_list = []

        self.N_DIIS = N_DIIS

    def prune_lists(self, fock_list, error_list):
        if ( len(fock_list) > self.N_DIIS ): # Remove if too long ago
            enorm = np.einsum("iab,iab->i", np.array(error_list), np.array(error_list))
            ind   = np.argmax( enorm )
            #print( np.round(enorm,2) )
            #print("Removing", ind)
            fock_list.pop(ind)
            error_list.pop(ind)
        return fock_list, error_list


    def __diis_extrapolate(self, F, error, fock_list, error_list):

        # Store Fock matrix
        fock_list.append( F )

        # Append new error vector
        error_list.append( error )
        
        # Prune lists
        fock_list, error_list = self.prune_lists( fock_list, error_list )

        N          = len( fock_list )
        rhs        = np.zeros(N + 1); rhs[-1] = 1
        B          = np.zeros((N + 1, N + 1))
        B[-1, :-1] = B[:-1, -1] = 1
        e          = np.array(error_list)
        B[:-1,:-1] = np.einsum("ixy,jxy->ij", e, e)
        coeff      = np.linalg.solve(B, rhs)[:-1]
        error_new  = np.einsum( "i,iab->ab", coeff, e )
        #print( np.round(np.linalg.norm(e),5) )
        #print( np.round(np.linalg.norm(error_new),5) )
        fock_new   = np.einsum( "i,iab->ab", coeff, fock_list )
        return fock_new, fock_list, error_list

    def diis_extrapolate(self, F_a, error_a, F_b=None, error_b=None):

        if ( F_b is None ):
            F_a, self.fock_list, self.error_list = self.__diis_extrapolate(F_a, error_a)
            return F_a
        else:
            if ( np.linalg.norm(self.error_list_a) < 1e-6 and np.linalg.norm(self.error_list_b) < 1e-6 ):
                return F_a, F_b
            F_a, self.fock_list_a, self.error_list_a = self.__diis_extrapolate(F_a, error_a, self.fock_list_a, self.error_list_a)
            F_b, self.fock_list_b, self.error_list_b = self.__diis_extrapolate(F_b, error_b, self.fock_list_b, self.error_list_b)
            return F_a, F_b

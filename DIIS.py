import numpy as np

class DIIS():
    def __init__( self, N_DIIS=8 ):
        self.fock_list   = []
        self.error_list  = []
        self.N_DIIS      = N_DIIS

    def __prune_lists(self):
        if ( len(self.fock_list) > self.N_DIIS ): # Remove if too long ago
            e = np.array(self.error_list)
            enorm = np.einsum("iab,iab->i", e, e)
            ind   = np.argmax( enorm )
            self.fock_list.pop(ind)
            self.error_list.pop(ind)

    def __get_error_vector(self, F, D):
        GRAD = D @ F
        return GRAD.conj().T - GRAD

    def __extrapolate(self, F, D):

        # Store Fock matrix
        self.fock_list.append( F )

        # Append new error vector
        error = self.__get_error_vector(F, D)
        self.error_list.append( error )

        #print( "\t\t   DIIS |error| = %1.1e" % np.linalg.norm(error) )
        
        # Prune lists
        self.__prune_lists()

        N          = len( self.fock_list )
        rhs        = np.zeros(N + 1); rhs[-1] = 1
        B          = np.zeros((N + 1, N + 1))
        B[-1, :-1] = B[:-1, -1] = 1
        e          = np.array(self.error_list)
        B[:-1,:-1] = np.einsum("ixy,jxy->ij", e, e)
        try:
            coeff      = np.linalg.solve(B, rhs)[:-1]
        except np.linalg.LinAlgError:
            print("   Warning! DIIS extrapolation failed. Setting Fock matrix and other variables to most recent.")
            self.error_list = None
            return self.fock_list[-1] # Reset variables
        #error_new  = np.einsum( "i,iab->ab", coeff, e )
        fock_new   = np.einsum( "i,iab->ab", coeff, self.fock_list )
        return fock_new

    def extrapolate(self, F, D):

        if ( self.error_list is None ): 
            return F
        else:
            return self.__extrapolate(F, D)

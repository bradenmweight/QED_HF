import numpy as np

class DIIS():
    def __init__( self, ao_overlap=None, unrestricted=False, N_DIIS=8 ):
        if ( unrestricted == True ):
            self.fock_list_a = []
            self.fock_list_b = []
            self.error_list_a = []
            self.error_list_b = []
        else:
            self.fock_list = []
            self.error_list = []

        self.N_DIIS = N_DIIS
        if ( ao_overlap is not None ):
            self.ao_overlap = ao_overlap

    def __prune_lists(self, fock_list, error_list):
        if ( len(fock_list) > self.N_DIIS ): # Remove if too long ago
            enorm = np.einsum("iab,iab->i", np.array(error_list), np.array(error_list))
            ind   = np.argmax( enorm )
            fock_list.pop(ind)
            error_list.pop(ind)
        return fock_list, error_list

    def __get_error_vector(self, F, D):
        if ( self.ao_overlap is not None ):
            GRAD = self.ao_overlap @ D @ F
        else:
            GRAD = D @ F
        return GRAD.conj().T - GRAD

    def __extrapolate(self, F, error, fock_list, error_list):

        # Store Fock matrix
        fock_list.append( F )

        # Append new error vector
        error_list.append( error )

        #print( "DIIS |error| = %1.2e" % np.linalg.norm(error) )
        
        # Prune lists
        fock_list, error_list = self.__prune_lists( fock_list, error_list )

        N          = len( fock_list )
        rhs        = np.zeros(N + 1); rhs[-1] = 1
        B          = np.zeros((N + 1, N + 1))
        B[-1, :-1] = B[:-1, -1] = 1
        e          = np.array(error_list)
        B[:-1,:-1] = np.einsum("ixy,jxy->ij", e, e)
        try:
            coeff      = np.linalg.solve(B, rhs)[:-1]
        except np.linalg.LinAlgError:
            print("   Warning! DIIS extrapolation failed. Setting Fock matrix and other variables to most recent.")
            #return fock_list[-1], [fock_list[-1]], [error_list[-1]] # Reset variables
            return fock_list[-1], None, None # Reset variables
        error_new  = np.einsum( "i,iab->ab", coeff, e )
        fock_new   = np.einsum( "i,iab->ab", coeff, fock_list )
        return fock_new, fock_list, error_list

    def extrapolate(self, F_a, D_a, F_b=None, D_b=None):

        if ( F_b is None ):
            if ( self.fock_list is None or self.error_list is None ):
                return F_a
            error_a = self.__get_error_vector(F_a, D_a)
            F_a, self.fock_list, self.error_list = self.__extrapolate(F_a, error_a, self.fock_list, self.error_list)
            return F_a
        else:
            if ( self.fock_list_a is None or self.fock_list_b is None ):
                return F_a, F_b
            error_a = self.__get_error_vector(F_a, D_a)
            error_b = self.__get_error_vector(F_b, D_b)
            F_a, self.fock_list_a, self.error_list_a = self.__extrapolate(F_a, error_a, self.fock_list_a, self.error_list_a)
            F_b, self.fock_list_b, self.error_list_b = self.__extrapolate(F_b, error_b, self.fock_list_b, self.error_list_b)
            return F_a, F_b

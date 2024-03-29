/*H********************************************************************************
* Ime datoteke: serverLinux.cpp
*
* Opis:
*		Enostaven strežnik, ki zmore sprejeti le enega klienta naenkrat.
*		Strežnik sprejme klientove podatke in jih v nespremenjeni obliki pošlje
*		nazaj klientu - odmev.
*
*H*/

//Vkljuèimo ustrezna zaglavja
#include<stdio.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<unistd.h>

#include <pthread.h>

/*
Definiramo vrata (port) na katerem bo strežnik poslušal
in velikost medponilnika za sprejemanje in pošiljanje podatkov
*/
#define PORT 8023
#define BUFFER_SIZE 256
#define MAX_CONNECTIONS 2

int CONNECTION_COUNT;
pthread_mutex_t LOCK;

// deklariram strukturo za prenašanje argumentov v niti
struct argumenti {
	int clientSock;         // ID odjemalčevega vtiča
};

struct argumenti args;

void *funkcijaNiti (void* arg);

int main(int argc, char **argv){
	CONNECTION_COUNT = 0;
	printf("[SERVER] Sever listening on port: %d\n", PORT);
	//Spremenjlivka za preverjane izhodnega statusa funkcij
	int iResult;
	/*
	Ustvarimo nov vtiè, ki bo poslušal
	in sprejemal nove kliente preko TCP/IP protokola
	*/
	int listener=socket(
		AF_INET,		// IPV4 domain 
		SOCK_STREAM, 	// TCP
		0);				// IP
	if (listener == -1) {
		printf("Error creating socket\n");
		return 1;
	}

	//Nastavimo vrata in mrežni naslov vtièa
	sockaddr_in  listenerConf;					// addrport structure
	listenerConf.sin_port=htons(PORT);			// Address port
	listenerConf.sin_family=AF_INET;			// Internet protocol (AF_INET)
	listenerConf.sin_addr.s_addr=INADDR_ANY;	// Internet address. or TCP/IPserver, internet address is usually settoI NADDR_ANY,i.e., any incoming interface

	// Vtiè povežemo z ustreznimi vrati
	// the bind function binds the socket to the address and port number specified in addr(custom data structure).
	iResult = bind( 
		listener, 					// socked id
		(sockaddr *)&listenerConf,  // addrport structure
		sizeof(listenerConf));		// the size (in bytes) of the addrport structure
	if (iResult == -1) {
		printf("Bind failed\n");
		close(listener);
		return 1;
    }

	//Zaènemo poslušati (listen for connections)
	if ( listen( 
			listener, 	// socket descriptor
			5			// # of active participants that can “wait” for a connectio
		) == -1 ) {
		printf( "Listen failed\n");
		close(listener);
		return 1;
	}

	//Definiramo nov vtiè in medpomnilik
	int clientSock;

    pthread_t nit;
	
	/*
	V zanki sprejemamo nove povezave
	in jih strežemo (najveè eno naenkrat)
	*/
	while (1)
	{
		//Sprejmi povezavo in ustvari nov vtiè
		// clientSock: the new socket (used for data-transfer)
		clientSock = accept(listener,NULL,NULL);
		if (clientSock == -1) {
			printf("Accept failed\n");
			close(listener);
			return 1;
		} else if (CONNECTION_COUNT == MAX_CONNECTIONS) {
			printf("[SERVER] Cannot accept more connections!\n");
			close(clientSock);
			continue;
		}

		// ko acceptas novo povezavo, ustvarti novo nit!
		// takoj ko jo ustvarite, jo detach!!!
        args.clientSock = clientSock;

		pthread_mutex_lock(&LOCK);
    	CONNECTION_COUNT++;
	    pthread_mutex_unlock(&LOCK);
		
        // ustvari niti:
		printf("[SERVER] New connection opened... We have: %d connections!\n", CONNECTION_COUNT);
        pthread_create( &nit,
                  NULL,
                  funkcijaNiti,
                  (void *) &args);


        pthread_detach(nit);

	}

	//Poèistimo vse vtièe
	close(listener);

	return 0;
}


void *funkcijaNiti (void* arg){
    struct argumenti *args ;
    args =  (struct argumenti *) arg;

    int clientSock = args->clientSock;
    int iResult;

    char buff[BUFFER_SIZE];


    // TOLE GRE V FUNKCIJO NITI, prenesi ji clientSock
	// 
	//Postrezi povezanemu klientu
	do{
		//Sprejmi podatke
		iResult = recv(clientSock, buff, BUFFER_SIZE, 0);
		if (iResult > 0) {
			printf("Bytes received: %d\n", iResult);

			//Vrni prejete podatke pošiljatelju
			iResult = send(clientSock, buff, iResult, 0 );
			if (iResult == -1) {
				printf("send failed!\n");
				close(clientSock);
				break;
			}
			printf("Bytes sent: %d\n", iResult);
		}
		else if (iResult == 0) {
			//printf("Connection closing...\n");
			pthread_mutex_lock(&LOCK);
    		CONNECTION_COUNT--;
	    	pthread_mutex_unlock(&LOCK);
			printf("[SERVER] Connection closing... We have: %d connections!\n", CONNECTION_COUNT);
		} else {
			printf("recv failed!\n");
			close(clientSock);
			break;
		}

	} while (iResult > 0);

	close(clientSock);

    pthread_exit(NULL);
}
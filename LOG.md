# Setup mongodb container
```dockerfile
services:
  mongodb:
    image: mongo:6.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      MONGO_INITDB_ROOT_USERNAME: root
      MONGO_INITDB_ROOT_PASSWORD: 123456@mongo
      MONGO_INITDB_DATABASE: chatbotdb
    networks:
      - mynetwork

networks:
  mynetwork:
    driver: bridge

volumes:
  mongodb_data:
```

docker ps

CONTAINER ID   IMAGE       COMMAND                  CREATED          STATUS          PORTS                                             NAMES
5ebbeb9e5c48   mongo:6.0   "docker-entrypoint.s…"   14 seconds ago   Up 14 seconds   0.0.0.0:27017->27017/tcp, [::]:27017->27017/tcp   multi_book_chatbot-mongodb-1

0.0.0.0:27017->27017/tcp → Your localhost:27017 is mapped to the container's 27017
[::]:27017->27017/tcp is the IPv6 mapping




curl -f http://localhost:9091/healthz 

mongosh "mongodb://root:123456@mongo@localhost:27017/chatbotdb?authSource=admin"


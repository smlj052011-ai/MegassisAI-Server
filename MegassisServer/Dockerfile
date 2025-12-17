FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["MegassisServer.csproj", "./"]
RUN dotnet restore "MegassisServer.csproj"
COPY . .
RUN dotnet publish "MegassisServer.csproj" -c Release -o /app/publish

FROM fedora:39
# Install Ollama
RUN curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama && \
    chmod +x /usr/bin/ollama

# Install ASP.NET Core Runtime
RUN dnf install -y aspnetcore-runtime-8.0 && dnf clean all

WORKDIR /app
COPY --from=build /app/publish .

EXPOSE 5087

# Create a script to start Ollama, pull a TINY model, and start the API
RUN echo '#!/bin/bash\nollama serve & sleep 5\nollama pull smollm:360m\ndotnet MegassisServer.dll' > /app/start.sh && \
    chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]
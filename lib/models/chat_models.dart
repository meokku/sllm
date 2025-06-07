// message 구조체 정의
class Message {
  final String content;
  final bool isUser;
  final DateTime timestamp;

  Message({
    required this.content,
    required this.isUser,
    required this.timestamp,
  });
}

// chat 구조체 정의
class Chat {
  final String id;
  final String title;
  final String lastMessage;
  final DateTime timestamp;
  final List<Message> messages;

  Chat({
    required this.id,
    required this.title,
    required this.lastMessage,
    required this.timestamp,
    required this.messages,
  });

  Chat copyWith({
    String? title,
    String? lastMessage,
    DateTime? timestamp,
    List<Message>? messages,
  }) {
    return Chat(
      id: id,
      title: title ?? this.title,
      lastMessage: lastMessage ?? this.lastMessage,
      timestamp: timestamp ?? this.timestamp,
      messages: messages ?? this.messages,
    );
  }
}

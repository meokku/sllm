import 'dart:async';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:swlab_sllm_app/models/chat_models.dart';
import 'package:swlab_sllm_app/services/firebase_service.dart';
import 'package:swlab_sllm_app/services/llm_service.dart';
import 'chat_session_provider.dart';

class ActiveChatProvider with ChangeNotifier {
  final LlmService _llmService;
  final ChatSessionProvider _chatSessionProvider;
  final FirebaseService _firebaseService;

  List<Message> _messages = [];
  bool _isLoading = false;
  StreamSubscription? _messagesStreamSubscription;

  ActiveChatProvider({
    required LlmService llmService,
    required ChatSessionProvider chatSessionProvider,
    required FirebaseService firebaseService,
  })  : _llmService = llmService,
        _chatSessionProvider = chatSessionProvider,
        _firebaseService = firebaseService {
    // 채팅 세션 변경 감지
    _chatSessionProvider.addListener(_onActiveChatChanged);
    _onActiveChatChanged();
  }

  List<Message> get messages => List.unmodifiable(_messages);
  bool get isLoading => _isLoading;

  void _onActiveChatChanged() {
    final activeChat = _chatSessionProvider.activeChat;
    print('Active chat changed: ${activeChat?.id}'); // 디버그 로그 추가

    // 메시지 스트림 구독 취소
    _messagesStreamSubscription?.cancel();

    // 메시지 목록 초기화
    _messages.clear();

    if (activeChat != null) {
      // 강제로 메시지 다시 로드
      _loadMessages(activeChat.id);
    } else {
      notifyListeners();
    }
  }

  void _loadMessages(String chatId) {
    print('Loading messages for chat: $chatId'); // 디버그 로그 추가

    // 이전 스트림 구독 취소
    _messagesStreamSubscription?.cancel();

    // 현재 사용자 ID 체크
    final currentUserId = _firebaseService.currentUserId;
    if (currentUserId == null) {
      print('No current user, cannot load messages');
      return;
    }

    // 새 스트림 구독
    _messagesStreamSubscription =
        _firebaseService.getChatMessages(chatId).listen(
      (snapshot) {
        print('Received ${snapshot.docs.length} messages'); // 디버그 로그 추가

        _messages = snapshot.docs.map((doc) {
          final data = doc.data() as Map<String, dynamic>;
          return Message(
            content: data['content'] ?? '',
            isUser: data['isUser'] ?? true,
            timestamp: data['timestamp'] != null
                ? (data['timestamp'] as Timestamp).toDate()
                : DateTime.now(),
          );
        }).toList();

        notifyListeners();
      },
      onError: (error) {
        print('Error loading messages: $error'); // 에러 로그 추가
      },
      cancelOnError: true,
    );
  }

  void setInitialMessage(String content) {
    // 사용자 메시지 객체 생성
    final userMessage = Message(
      content: content,
      isUser: true,
      timestamp: DateTime.now(),
    );

    // 메시지 목록에 추가
    _messages.add(userMessage);
    notifyListeners();

    // 로딩 상태 설정 (UI에 로딩 표시됨)
    _isLoading = true;
    notifyListeners();
  }

  Future<void> sendMessage(String text) async {
    if (text.trim().isEmpty) return;

    // 활성화된 채팅이 없으면 새로 만듦
    if (_chatSessionProvider.activeChat == null) {
      await _chatSessionProvider.createNewChat();
    }

    final activeChat = _chatSessionProvider.activeChat!;
    final userMessage = Message(
      content: text,
      isUser: true,
      timestamp: DateTime.now(),
    );

    _isLoading = true;
    notifyListeners();

    try {
      // 사용자 메시지 저장
      await _firebaseService.saveMessage(activeChat.id, userMessage);

      // // AI 응답 요청
      // final response = await _llmService.askLlama(text);

      // // AI 메시지 저장
      // final aiMessage = Message(
      //   content: response,
      //   isUser: false,
      //   timestamp: DateTime.now(),
      // );
      // AI 메시지 초기 생성 (빈 텍스트로 먼저 추가)
      var aiContent = '';
      final aiMessage = Message(
        content: aiContent,
        isUser: false,
        timestamp: DateTime.now(),
      );
      _messages.add(aiMessage);
      notifyListeners();

      // AI 응답 Streaming 요청
      final stream = _llmService.askLlamaStream(text);

      // 스트리밍 응답 처리
      await for (final chunk in stream) {
        aiContent += chunk;

        // 리스트에서 기존 메시지 위치 찾기
        final index = _messages.indexOf(aiMessage);
        if (index != -1) {
          // 새 Message 객체로 교체
          _messages[index] = Message(
            content: aiContent,
            isUser: false,
            timestamp: aiMessage.timestamp,
          );
          notifyListeners(); // 매번 업데이트해서 UI 실시간 갱신
        }
      }

      // 최종 AI 메시지 저장
      await _firebaseService.saveMessage(activeChat.id, aiMessage);
    } catch (e) {
      // 오류 메시지 저장
      final errorMessage = Message(
        content: '오류 발생: $e',
        isUser: false,
        timestamp: DateTime.now(),
      );

      await _firebaseService.saveMessage(activeChat.id, errorMessage);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void clearCurrentChat() {
    // Firebase에서는 실제 메시지를 삭제하는 대신 새 채팅을 만드는 것이 좋음
    _chatSessionProvider.createNewChat();
  }

  void clearActiveChat() {
    _messagesStreamSubscription?.cancel();
    _messages.clear();
    _isLoading = false;
    _messagesStreamSubscription = null;
    notifyListeners();
  }

  @override
  void dispose() {
    _chatSessionProvider.removeListener(_onActiveChatChanged);
    _messagesStreamSubscription?.cancel();
    super.dispose();
  }
}

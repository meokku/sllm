import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:swlab_sllm_app/models/chat_models.dart';
import 'package:swlab_sllm_app/providers/active_chat_provider.dart';
import 'package:swlab_sllm_app/providers/chat_session_provider.dart';
import 'package:swlab_sllm_app/services/auth_service.dart';
import 'package:swlab_sllm_app/theme/colors.dart';
import 'package:swlab_sllm_app/utils/profile_menu.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> with TickerProviderStateMixin {
  final AuthService _authService = AuthService();
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  late AnimationController _dotAnimationController;
  int _currentDotCount = 1;
  Timer? _dotAnimationTimer;

  // 프로필 메뉴 관련 변수
  final GlobalKey _profileButtonKey = GlobalKey();
  OverlayEntry? _overlayEntry;

  // 왼쪽 사이드바 관련 변수
  bool _isSidebarExpanded = true;
  late AnimationController _sidebarAnimationController;
  late Animation<double> _sidebarAnimation;

  @override
  void initState() {
    super.initState();
    _dotAnimationController = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: 500),
    );

    // 사이드바 애니메이션 컨트롤러 초기화
    _sidebarAnimationController = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: 300),
    );

    // 사이드바 너비 애니메이션 (200 <-> 0)
    _sidebarAnimation = Tween<double>(
      begin: 200.0,
      end: 0,
    ).animate(CurvedAnimation(
      parent: _sidebarAnimationController,
      curve: Curves.easeInOut,
    ));

    // 점(...) 애니메이션을 위한 타이머 설정
    _dotAnimationTimer = Timer.periodic(Duration(milliseconds: 500), (timer) {
      setState(() {
        _currentDotCount = (_currentDotCount % 3) + 1;
      });
    });
  }

  @override
  void dispose() {
    _textController.dispose();
    _scrollController.dispose();
    _dotAnimationController.dispose();
    _dotAnimationTimer?.cancel();
    _sidebarAnimationController.dispose();
    _removeOverlay();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // 사이드바 토글 함수
  void _toggleSidebar() {
    setState(() {
      _isSidebarExpanded = !_isSidebarExpanded;
      if (_isSidebarExpanded) {
        _sidebarAnimationController.reverse();
      } else {
        _sidebarAnimationController.forward();
      }
    });
  }

  void _showProfileMenu() {
    // 현재 오버레이가 표시 중이면 제거
    _removeOverlay();

    // ProfileMenuUtil 사용
    ProfileMenuUtil.showProfileMenu(
      context,
      buttonKey: _profileButtonKey,
      onRemove: _removeOverlay,
      onOverlayCreated: (entry) {
        _overlayEntry = entry;
      },
    );
  }

  void _removeOverlay() {
    _overlayEntry?.remove();
    _overlayEntry = null;
  }

  @override
  Widget build(BuildContext context) {
    final chatSessionProvider = Provider.of<ChatSessionProvider>(context);
    final activeChatProvider = Provider.of<ActiveChatProvider>(context);
    final user = _authService.currentUser;

    // 새 메시지가 추가되면 스크롤
    if (activeChatProvider.messages.isNotEmpty) {
      _scrollToBottom();
    }

    return Scaffold(
      body: Stack(children: [
        AnimatedBuilder(
            animation: _sidebarAnimation,
            builder: (context, child) {
              // 사이드바가 일정 너비 이하로 줄어들면 내부 콘텐츠를 숨김
              final bool showContent = _sidebarAnimation.value > 120; // 임계값 설정

              return Container(
                width: _sidebarAnimation.value,
                color: Colors.grey[100],
                child: Column(
                  children: [
                    // 로고 자리 공간 확보 (투명)
                    SizedBox(height: 60),
                    if (_isSidebarExpanded && showContent) ...[
                      SizedBox(
                        height: 25,
                      ),
                      // 새 채팅 버튼
                      Material(
                        color: Colors.transparent, // 배경색 투명하게 설정
                        child: InkWell(
                          onTap: () {
                            print('새 채팅 버튼이 클릭되었습니다');
                            Navigator.pushReplacementNamed(
                                context, '/'); // 화면 교체
                          },
                          child: Padding(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 12.0, vertical: 10),
                            child: Row(
                              children: [
                                Container(
                                  width: 24,
                                  height: 24,
                                  decoration: BoxDecoration(
                                    color: SKKUColors.green,
                                    shape: BoxShape.circle,
                                  ),
                                  child: Center(
                                    child: Icon(
                                      Icons.add,
                                      color: Colors.white,
                                      size: 16,
                                    ),
                                  ),
                                ),
                                SizedBox(width: 8),
                                Text(
                                  "새 채팅",
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                      SizedBox(
                        height: 30,
                      ),
                      // 최근 항목 텍스트
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 12.0),
                        child: Align(
                          alignment: Alignment.centerLeft,
                          child: Text(
                            "최근 항목",
                            style: TextStyle(
                              color: Colors.grey[600],
                              fontSize: 14,
                            ),
                          ),
                        ),
                      ),
                      SizedBox(
                        height: 10,
                      ),
                      // 채팅 목록
                      Expanded(
                        child: chatSessionProvider.isLoading
                            ? Center(child: CircularProgressIndicator())
                            : Builder(builder: (context) {
                                // 화면 build 시 복사본 준비 (Stack 복사본)
                                List<Chat> stackCopy =
                                    List.from(chatSessionProvider.chatSessions);

                                return ListView.builder(
                                  itemCount: stackCopy.length,
                                  itemBuilder: (context, index) {
                                    // Stack pop → LIFO 시연
                                    final chat = stackCopy.removeLast();

                                    return Material(
                                      color: Colors.grey[100],
                                      child: InkWell(
                                        onTap: () {
                                          // 클릭 시 → 원본 _chatSessions 기준으로 selectChat
                                          chatSessionProvider
                                              .selectChat(chat.id);
                                          Navigator.pushReplacementNamed(
                                              context, '/chat');
                                        },
                                        child: Container(
                                          width: double.infinity,
                                          padding: EdgeInsets.symmetric(
                                              vertical: 8, horizontal: 12),
                                          child: Text(
                                            chat.title,
                                            style: TextStyle(fontSize: 14),
                                          ),
                                        ),
                                      ),
                                    );
                                  },
                                );
                              }),
                      ),
                    ]
                  ],
                ),
              );
            }),

        // 채팅 영역
        AnimatedContainer(
          duration: Duration(milliseconds: 300),
          curve: Curves.easeInOut,
          margin: EdgeInsets.only(
              left: _isSidebarExpanded ? 200 : 0), // 왼쪽 마진으로 위치 조정
          child: Center(
            child: Container(
              width: 1000, // 채팅 영역 최대 너비 제한
              padding: EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                children: [
                  // 메시지 목록
                  Expanded(
                    child: ListView.builder(
                      controller: _scrollController,
                      padding: EdgeInsets.all(16.0),
                      itemCount: activeChatProvider.isLoading
                          ? activeChatProvider.messages.length + 1
                          : activeChatProvider.messages.length,
                      itemBuilder: (context, index) {
                        // 마지막 항목이고 로딩 중이라면 로딩 표시기 반환
                        if (activeChatProvider.isLoading &&
                            index == activeChatProvider.messages.length) {
                          return _buildLoadingText();
                        }
                        final message = activeChatProvider.messages[index];
                        return _buildMessageBubble(message);
                      },
                    ),
                  ),

                  // 메시지 입력 영역
                  Container(
                    padding: EdgeInsets.all(16.0),
                    child: Row(
                      children: [
                        Expanded(
                          child: Container(
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(24.0),
                              border: Border.all(color: Colors.grey[300]!),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(0.05),
                                  blurRadius: 2,
                                  spreadRadius: 1,
                                ),
                              ],
                            ),
                            child: Column(
                              children: [
                                Row(
                                  children: [
                                    Expanded(
                                      child: TextField(
                                        maxLines: null,
                                        textInputAction: TextInputAction.send,
                                        keyboardType: TextInputType.text,
                                        controller: _textController,
                                        decoration: InputDecoration(
                                          hintText: '무엇을 도와드릴까요?',
                                          hintStyle: TextStyle(
                                            color: Colors.grey,
                                          ),
                                          border: InputBorder.none,
                                          contentPadding: EdgeInsets.symmetric(
                                              horizontal: 20.0),
                                        ),
                                        onSubmitted: (text) {
                                          if (!activeChatProvider.isLoading) {
                                            activeChatProvider
                                                .sendMessage(text);
                                            _textController.clear();
                                          }
                                        },
                                      ),
                                    ),
                                    Container(
                                      margin:
                                          EdgeInsets.only(right: 10, top: 10),
                                      decoration: BoxDecoration(
                                        color: SKKUColors.green,
                                        shape: BoxShape.circle,
                                      ),
                                      child: IconButton(
                                        icon: Icon(Icons.arrow_upward,
                                            color: Colors.white),
                                        onPressed: () {
                                          if (!activeChatProvider.isLoading) {
                                            activeChatProvider.sendMessage(
                                                _textController.text);
                                            _textController.clear();
                                          }
                                        },
                                      ),
                                    ),
                                  ],
                                ),
                                // 추가 입력 옵션 (+ 버튼)
                                Container(
                                  alignment: Alignment.centerLeft,
                                  padding:
                                      EdgeInsets.only(left: 10.0, bottom: 10.0),
                                  child: Container(
                                    width: 32,
                                    height: 32,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      border:
                                          Border.all(color: Colors.grey[300]!),
                                    ),
                                    child: IconButton(
                                      icon: Icon(Icons.add,
                                          size: 20, color: Colors.grey),
                                      onPressed: () {
                                        // 추가 버튼 클릭 시
                                        // 파일 추가 등의 로직 설정
                                      },
                                      padding: EdgeInsets.zero,
                                      constraints: BoxConstraints(),
                                    ),
                                  ),
                                )
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),

        // 로고 및 토글 버튼 (위 레이어)
        Positioned(
          top: 0,
          left: 0,
          height: 60,
          width: 200,
          child: SizedBox(
            child: Padding(
              padding: const EdgeInsets.only(left: 15.0, top: 13, right: 5),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Image.asset(
                    "assets/images/SKKULLM.png",
                    width: 120,
                  ),
                  IconButton(
                    onPressed: _toggleSidebar,
                    icon: Icon(_isSidebarExpanded
                        ? Icons.keyboard_double_arrow_left_rounded
                        : Icons.keyboard_double_arrow_right_rounded),
                    tooltip: _isSidebarExpanded ? "사이드바 접기" : "사이드바 펼치기",
                  ),
                ],
              ),
            ),
          ),
        ),
        // 유저 프로필 버튼
        Positioned(
          top: 0,
          right: 0,
          height: 60,
          width: 60,
          child: Padding(
            padding: const EdgeInsets.only(right: 15.0, top: 13, left: 5),
            child: Tooltip(
              message: "사용자 프로필",
              child: Material(
                key: _profileButtonKey,
                color: Colors.transparent,
                child: Ink(
                  decoration: BoxDecoration(
                    color: Colors.grey[400],
                    shape: BoxShape.circle,
                  ),
                  child: InkWell(
                    customBorder: CircleBorder(),
                    onTap: () {
                      _showProfileMenu();
                      print(user?.email);
                    },
                    child: SizedBox(
                      width: 40,
                      height: 40,
                      child: Icon(
                        Icons.person,
                        color: Colors.white,
                        size: 30,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ]),
    );
  }

  // 로딩 텍스트 위젯
  Widget _buildLoadingText() {
    String dots = '.' * _currentDotCount;
    return Align(
      alignment: Alignment.centerLeft,
      child: Padding(
        padding: EdgeInsets.symmetric(vertical: 16.0, horizontal: 24.0),
        child: Text(
          "채팅 생성 중$dots",
          style: TextStyle(
            color: Colors.grey[700],
            fontSize: 15,
          ),
        ),
      ),
    );
  }

  Widget _buildMessageBubble(Message message) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: EdgeInsets.symmetric(vertical: 8.0, horizontal: 12.0),
        padding: EdgeInsets.symmetric(vertical: 12.0, horizontal: 16.0),
        decoration: BoxDecoration(
          color: message.isUser ? SKKUColors.green : Colors.grey[200],
          borderRadius: BorderRadius.circular(20.0),
        ),
        constraints: BoxConstraints(maxWidth: 600),
        child: Text(
          message.content,
          style: TextStyle(
            color: message.isUser ? Colors.white : Colors.black87,
            fontSize: 15,
          ),
        ),
      ),
    );
  }
}

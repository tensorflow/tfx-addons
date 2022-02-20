# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: slack.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
    name='slack.proto',
    package='tfx_addons.slack_exit_handler',
    syntax='proto3',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x0bslack.proto\x12\x1dtfx_addons.slack_exit_handler\":\n\tSlackSpec\x12\x13\n\x0bslack_token\x18\x01 \x01(\t\x12\x18\n\x10slack_channel_id\x18\x02 \x01(\tb\x06proto3'  # pylint: disable=line-too-long
)




_SLACKSPEC = _descriptor.Descriptor(
    name='SlackSpec',
    full_name='tfx_addons.slack_exit_handler.SlackSpec',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,  # pylint: disable=protected-access
    fields=[
      _descriptor.FieldDescriptor(
          name='slack_token', full_name='tfx_addons.slack_exit_handler.SlackSpec.slack_token', index=0,  # pylint: disable=line-too-long
          number=1, type=9, cpp_type=9, label=1,
          has_default_value=False, default_value=b"".decode('utf-8'),
          message_type=None, enum_type=None, containing_type=None,
          is_extension=False, extension_scope=None,
          serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),  # pylint: disable=line-too-long
      _descriptor.FieldDescriptor(
          name='slack_channel_id', full_name='tfx_addons.slack_exit_handler.SlackSpec.slack_channel_id', index=1,  # pylint: disable=line-too-long
          number=2, type=9, cpp_type=9, label=1,
          has_default_value=False, default_value=b"".decode('utf-8'),
          message_type=None, enum_type=None, containing_type=None,
          is_extension=False, extension_scope=None,
          serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),  # pylint: disable=protected-access
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=104,
)

DESCRIPTOR.message_types_by_name['SlackSpec'] = _SLACKSPEC
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SlackSpec = _reflection.GeneratedProtocolMessageType(
    'SlackSpec',
    (_message.Message,),
    {
        'DESCRIPTOR' : _SLACKSPEC,
        '__module__' : 'slack_pb2'
        # @@protoc_insertion_point(class_scope:tfx_addons.slack_exit_handler.SlackSpec)
    })
_sym_db.RegisterMessage(SlackSpec)


# @@protoc_insertion_point(module_scope)
